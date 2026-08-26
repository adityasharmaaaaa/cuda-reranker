#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 128

__global__ void attention_vectorized_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq, int head_dim
){
    int batch_head = blockIdx.x;
    int query_idx = threadIdx.x;
    if (query_idx >= seq) return;

    const float* q_row  = Q + (size_t)batch_head * seq * head_dim + query_idx * head_dim;
    const float* k_base = K + (size_t)batch_head * seq * head_dim;
    const float* v_base = V + (size_t)batch_head * seq * head_dim;
    float*       o_row  = O + (size_t)batch_head * seq * head_dim + query_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    alignas(16) float q[32];
    
    // We can also vectorize the local Q load since PyTorch tensor rows are 
    // heavily aligned, but we'll stick to scalar to keep focus on the inner loop.
    for (int d = 0; d < head_dim; d++) q[d] = q_row[d];

    // Force alignment on shared memory
    __shared__ alignas(16) float tile_K[TILE * 32];
    __shared__ alignas(16) float tile_V[TILE * 32];

    float scores[1024];

    // Pre-cast our query register array to a float4 pointer
    const float4* q_vec = reinterpret_cast<const float4*>(q);

    for (int tile_start = 0; tile_start < seq; tile_start += TILE) {
        int tile_len = min(TILE, seq - tile_start);
        int total_elements = tile_len * head_dim;
        
        // Cooperative load: left as perfectly coalesced scalar loads
        for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
            tile_K[i] = k_base[tile_start * head_dim + i];
            tile_V[i] = v_base[tile_start * head_dim + i];
        }
        __syncthreads();
        
        for (int j = 0; j < tile_len; j++) {
            float dot = 0.0f;
            // Point a float4 pointer to the start of the current K row in shared memory
            const float4* k_vec = reinterpret_cast<const float4*>(&tile_K[j * head_dim]);
            
            // Loop runs head_dim / 4 times (8 iterations instead of 32)
            for (int d = 0; d < head_dim / 4; d++) {
                float4 q4 = q_vec[d];
                float4 k4 = k_vec[d];
                dot += (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            }
            scores[tile_start + j] = dot * scale;
        }
        __syncthreads();
    }

    float maxi = -1e20f;
    for (int i = 0; i < seq; i++) {
        maxi = fmaxf(maxi, scores[i]);
    }

    float sum = 0.0f;
    for (int i = 0; i < seq; i++) {
        float x = expf(scores[i] - maxi);
        sum += x;
        scores[i] = x;
    }

    for (int i = 0; i < seq; i++) {
        scores[i] = scores[i] / sum;
    }

    alignas(16) float acc[32] = {0};
    float4* acc_vec = reinterpret_cast<float4*>(acc);

    for (int tile_start = 0; tile_start < seq; tile_start += TILE) {
        int tile_len = min(TILE, seq - tile_start);
        int total_elements = tile_len * head_dim;
        
        for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
            tile_V[i] = v_base[tile_start * head_dim + i];
        }
        __syncthreads();
        
        for (int j = 0; j < tile_len; j++) {
            float s = scores[tile_start + j];
            const float4* v_vec = reinterpret_cast<const float4*>(&tile_V[j * head_dim]);
            
            for (int dim = 0; dim < head_dim / 4; dim++) {
                float4 v4 = v_vec[dim];
                acc_vec[dim].x += s * v4.x;
                acc_vec[dim].y += s * v4.y;
                acc_vec[dim].z += s * v4.z;
                acc_vec[dim].w += s * v4.w;
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int dim = 0; dim < head_dim; dim++) {
        o_row[dim] = acc[dim];
    }
}

torch::Tensor attention_vectorized_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "fp32 only for now");
    
    auto sizes = Q.sizes();
    int batch = sizes[0], heads = sizes[1], seq = sizes[2], head_dim = sizes[3];
    
    // Safety check: head_dim must be divisible by 4 for our float4 logic
    TORCH_CHECK(head_dim % 4 == 0, "head_dim must be a multiple of 4 for float4 vectorization");
    
    auto O = torch::empty_like(Q);
    
    dim3 grid(batch * heads);
    dim3 block(seq);
    
    attention_vectorized_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), seq, head_dim
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_vectorized_forward, "vectorized tiled attention forward (CUDA)");
}