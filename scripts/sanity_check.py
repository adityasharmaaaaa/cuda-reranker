import torch
from torch.utils.cpp_extension import load_inline

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))

cuda_src = r"""
__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""
cpp_src = "torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);"

mod = load_inline(
    name="sanity_add",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["add_cuda"],
    verbose=True,
)

a = torch.randn(1024, device="cuda")
b = torch.randn(1024, device="cuda")
out = mod.add_cuda(a, b)
max_err = (out - (a + b)).abs().max().item()
print("max error vs PyTorch:", max_err)
assert max_err < 1e-5, "sanity kernel FAILED"
print("Sanity check PASSED — toolchain is working end to end.")