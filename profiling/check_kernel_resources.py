from torch.utils.cpp_extension import load

load(
    name="attention_naive_verbose",
    sources=["cuda/attention_naive.cu"],
    extra_cuda_cflags=["-Xptxas","-v"],
    verbose=True
)