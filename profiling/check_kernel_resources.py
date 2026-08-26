from torch.utils.cpp_extension import load

load(
    name="attention_vectorized_verbose",
    sources=["cuda/attention_vectorized.cu"],
    extra_cuda_cflags=["-Xptxas","-v"],
    verbose=True
)