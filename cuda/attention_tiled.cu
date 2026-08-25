#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE 128

__global__ void attention_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq, int head_dim
){
    int batch_head=blockIdx.x;
    int query_idx=threadIdx.x;
    if(query_idx>=seq) return ;

    const float* q_row  = Q + (size_t)batch_head * seq * head_dim + query_idx * head_dim;
    const float* k_base = K + (size_t)batch_head * seq * head_dim;
    const float* v_base = V + (size_t)batch_head * seq * head_dim;
    float*       o_row  = O + (size_t)batch_head * seq * head_dim + query_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    float q[32];
    for (int d = 0; d < head_dim; d++) q[d] = q_row[d];

    __shared__ float tile_K[TILE * 32];
    __shared__ float tile_V[TILE * 32];

    float scores[1024];

    for(int tile_start=0; tile_start<seq; tile_start+=TILE){
        int tile_len=min(TILE,seq-tile_start);
        int total_elements=tile_len*head_dim;
        for(int i=threadIdx.x; i<total_elements; i+=blockDim.x){
            tile_K[i]=k_base[tile_start*head_dim+i];
            tile_V[i]=v_base[tile_start*head_dim+i];
        }
        __syncthreads();
        for(int j=0; j<tile_len; j++){
            float dot=0.0f;
            for(int d=0; d<head_dim; d++){
                dot+=q[d]*tile_K[j*head_dim+d];
            }
            scores[tile_start+j]=dot*scale;
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

    float acc[32] = {0};

    for(int tile_start = 0; tile_start < seq; tile_start += TILE){
        int tile_len = min(TILE, seq - tile_start);
        int total_elements = tile_len * head_dim;
        
        for(int i = threadIdx.x; i < total_elements; i += blockDim.x){
            tile_V[i] = v_base[tile_start * head_dim + i];
        }
        __syncthreads();
        
        for (int j = 0; j < tile_len; j++) {
            float s = scores[tile_start + j];
            for (int dim = 0; dim < head_dim; dim++) {
                acc[dim] += s * tile_V[j * head_dim + dim];
            }
        }
        __syncthreads();
    }

    for (int dim = 0; dim < head_dim; dim++) {
        o_row[dim] = acc[dim];
    }
}

torch::Tensor attention_tiled_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "fp32 only for now");
    auto sizes = Q.sizes();
    int batch = sizes[0], heads = sizes[1], seq = sizes[2], head_dim = sizes[3];
    auto O = torch::empty_like(Q);
    dim3 grid(batch * heads);
    dim3 block(seq);
    attention_tiled_kernel<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), O.data_ptr<float>(), seq, head_dim
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &attention_tiled_forward, "tiled attention forward (CUDA)");
}