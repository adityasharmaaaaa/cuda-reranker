#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void attention_naive_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq, int head_dim
){
    int batch_head=blockIdx.x;
    int query_idx=threadIdx.x;
    if(query_idx>=seq) return;

    const float* q_row = Q + (size_t)batch_head * seq * head_dim + query_idx * head_dim;
    const float* k_base = K + (size_t)batch_head * seq * head_dim;
    const float* v_base = V + (size_t)batch_head * seq * head_dim;
    float* o_row = O + (size_t)batch_head * seq * head_dim + query_idx * head_dim;

    float scale = 1.0f / sqrtf((float)head_dim);

    float scores[seq];
    for(int j=0; j<seq; j++){
        float dot_product=0.0f;
        for(int d=0; d<head_dim; d++){
            dot_product+=q_row[d]*(k_base+j*head_dim)[d];
        }
        scores[j]=scale*dot_product;
    }
   
    float maxi=-INFINITY;
    for(int i=0; i<seq; i++){
        maxi=fmaxf(maxi,scores[i]);
    }
    float sum=0;
    for(int i=0; i<seq; i++){
        float x=expf(scores[i]-maxi);
        sum+=x;
        scores[i]=x;
    }
    for(int i=0; i<seq; i++){
        scores[i]=scores[i]/sum;
    }

    for(int dim=0; dim<head_dim; dim++){
        float values=0.0f;
        for(int j=0; j<seq; j++){
            values+=scores[j]*v_base[j*head_dim+dim];
        }
        o_row[d]=values;
    }
}

torch::Tensor attention_naive_forward(torch:: Tensor Q, torch::Tensor K, torch::Tensor V){
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "fp32 only for now");

    auto sizes = Q.sizes();
    int batch = sizes[0], heads = sizes[1], seq = sizes[2], head_dim=sizes[3];
    auto O = torch::empty_like(Q);

    dim3 grid(batch*heads);
    dim3 block(seq);

    attention_naive_kernel<<<grid,block>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        seq, head_dim
    );
    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward",&attention_naive_forward, "naive attention forward (CUDA)")
}