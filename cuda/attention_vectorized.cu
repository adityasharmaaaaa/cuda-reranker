#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_LEN 32 
#define HEAD_DIM 32 

__global__ void attention_vectorized_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int* __restrict__ valid_lengths,  
    int batch_size,
    int num_heads,
    int seq_len,
    float scale
) {
    int batch_idx = blockIdx.y;
    int head_idx = blockIdx.z;
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int valid_len = valid_lengths[batch_idx];

    // Calculate base pointers for the specific batch and head
    int head_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const float* q_base = q + head_offset;
    const float* k_base = k + head_offset;
    const float* v_base = v + head_offset;
    float* out_base = out + head_offset;

    // Dynamic shared memory for cooperative K & V tiles
    extern __shared__ float sram[];
    float* tile_K = sram;
    float* tile_V = sram + (TILE_LEN * HEAD_DIM);

    // 16-byte Aligned Thread-local registers
    alignas(16) float q_reg[HEAD_DIM];
    alignas(16) float acc[HEAD_DIM] = {0.0f};
    
    float4* q_vec = reinterpret_cast<float4*>(q_reg);
    float4* acc_vec = reinterpret_cast<float4*>(acc);

    // Vectorized load Q into registers for this thread
    if (q_idx < seq_len) {
        const float4* q_global_vec = reinterpret_cast<const float4*>(&q_base[q_idx * HEAD_DIM]);
        #pragma unroll
        for (int d = 0; d < HEAD_DIM / 4; d++) {
            q_vec[d] = q_global_vec[d];
        }
    }

    float max_logit = -INFINITY;
    float sum_exp = 0.0f;

    int num_tiles = (seq_len + TILE_LEN - 1) / TILE_LEN;
    int total_elements = TILE_LEN * HEAD_DIM;
    
    // Number of float4 vectors to load cooperatively
    int total_vec_elements = total_elements / 4;

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_start = tile * TILE_LEN;
        
        // 1. Vectorized Cooperative Tile Load
        const float4* k_base_vec = reinterpret_cast<const float4*>(k_base + tile_start * HEAD_DIM);
        const float4* v_base_vec = reinterpret_cast<const float4*>(v_base + tile_start * HEAD_DIM);
        float4* tile_K_vec = reinterpret_cast<float4*>(tile_K);
        float4* tile_V_vec = reinterpret_cast<float4*>(tile_V);

        for (int i = threadIdx.x; i < total_vec_elements; i += blockDim.x) {
            int row_in_tile = i / (HEAD_DIM / 4);
            int global_row = tile_start + row_in_tile;
            if (global_row < seq_len) {
                tile_K_vec[i] = k_base_vec[i];
                tile_V_vec[i] = v_base_vec[i];
            }
        }
        __syncthreads();

        if (q_idx < seq_len) {
            int keys_in_tile = min(TILE_LEN, seq_len - tile_start);
            float scores[TILE_LEN];

            // 2. Vectorized Inner Loop 1: Q * K^T
            for (int k_idx = 0; k_idx < keys_in_tile; k_idx++) {
                float4* k_local_vec = reinterpret_cast<float4*>(&tile_K[k_idx * HEAD_DIM]);
                float score = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    float4 q_v = q_vec[d];
                    float4 k_v = k_local_vec[d];
                    score += q_v.x * k_v.x + q_v.y * k_v.y + q_v.z * k_v.z + q_v.w * k_v.w;
                }
                
                int global_k_idx = tile_start + k_idx;
                if (global_k_idx >= valid_len) {
                    scores[k_idx] = -INFINITY;
                } else {
                    scores[k_idx] = score * scale;
                }
            }

            float old_max = max_logit;
            for (int k_idx = 0; k_idx < keys_in_tile; k_idx++) {
                max_logit = fmaxf(max_logit, scores[k_idx]);
            }

            // GUARD: Prevent expf(-INF - (-INF)) which yields NaN.
            // If max_logit is still -INF, there have been no valid keys yet.
            float exp_correction = (max_logit == -INFINITY) ? 0.0f : expf(old_max - max_logit);
            sum_exp *= exp_correction;
            
            // Apply scale to existing accumulator
            #pragma unroll
            for (int d = 0; d < HEAD_DIM / 4; d++) {
                acc_vec[d].x *= exp_correction;
                acc_vec[d].y *= exp_correction;
                acc_vec[d].z *= exp_correction;
                acc_vec[d].w *= exp_correction;
            }

            // 3. Vectorized Inner Loop 2: Softmax * V
            for (int k_idx = 0; k_idx < keys_in_tile; k_idx++) {
                // GUARD: Prevent NaN when scores[k_idx] and max_logit are both -INF
                float exp_score = (max_logit == -INFINITY) ? 0.0f : expf(scores[k_idx] - max_logit);
                sum_exp += exp_score;

                float4* v_local_vec = reinterpret_cast<float4*>(&tile_V[k_idx * HEAD_DIM]);
                #pragma unroll
                for (int d = 0; d < HEAD_DIM / 4; d++) {
                    float4 v_v = v_local_vec[d];
                    acc_vec[d].x += exp_score * v_v.x;
                    acc_vec[d].y += exp_score * v_v.y;
                    acc_vec[d].z += exp_score * v_v.z;
                    acc_vec[d].w += exp_score * v_v.w;
                }
            }
        }
        __syncthreads();
    }

    if (q_idx < seq_len) {
        // GUARD: If a query sequence is completely masked, sum_exp will be 0.0f.
        // Prevent div-by-zero turning 0.0f acc into NaN.
        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM / 4; d++) {
            acc_vec[d].x *= inv_sum;
            acc_vec[d].y *= inv_sum;
            acc_vec[d].z *= inv_sum;
            acc_vec[d].w *= inv_sum;
        }

        // 4. Vectorized Global Memory Write
        float* o_row = out_base + q_idx * HEAD_DIM;
        float4* o_row_vec = reinterpret_cast<float4*>(o_row);
        
        #pragma unroll
        for (int v = 0; v < HEAD_DIM / 4; v++) {
            o_row_vec[v] = acc_vec[v];
        }
    }
}

torch::Tensor forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor valid_lengths) {
    TORCH_CHECK(q.size(3) == HEAD_DIM, "kernel hardcodes head_dim=32, got ", q.size(3));
    
    // Validate valid_lengths tensor
    TORCH_CHECK(valid_lengths.dim() == 1, "valid_lengths must be a 1D tensor");
    TORCH_CHECK(valid_lengths.size(0) == q.size(0), "valid_lengths must match batch_size");
    TORCH_CHECK(valid_lengths.scalar_type() == torch::kInt32, "valid_lengths must be int32");
    TORCH_CHECK(valid_lengths.is_cuda(), "valid_lengths must be on CUDA");

    int batch_size = q.size(0);
    int num_heads = q.size(1);
    int seq_len = q.size(2);
    // head_dim inferred as HEAD_DIM (32) internally

    auto out = torch::empty_like(q);

    dim3 threads_per_block(TILE_LEN);
    dim3 blocks_per_grid((seq_len + TILE_LEN - 1) / TILE_LEN, batch_size, num_heads);

    // K & V tiles: 2 * TILE_LEN * HEAD_DIM elements
    size_t smem_size = 2 * TILE_LEN * HEAD_DIM * sizeof(float);
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    attention_vectorized_kernel<<<blocks_per_grid, threads_per_block, smem_size>>>(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        valid_lengths.data_ptr<int>(), // Pass through memory pointer
        batch_size,
        num_heads,
        seq_len,
        scale
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Tiled Attention");
}