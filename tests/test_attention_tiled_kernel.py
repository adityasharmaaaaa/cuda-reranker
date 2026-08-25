import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

attention_tiled = load(name="attention_tiled", sources=["cuda/attention_tiled.cu"], verbose=True)

def test_attention_tiled_matches_reference():
    torch.manual_seed(0)
    batch,heads,seq,head_dim=2,4,16,32
    q=torch.randn(batch,heads,seq,head_dim,device="cuda",dtype=torch.float32)
    k=torch.randn(batch,heads,seq,head_dim,device="cuda",dtype=torch.float32)
    v=torch.randn(batch,heads,seq,head_dim,device="cuda",dtype=torch.float32)

    ref=F.scaled_dot_product_attention(q,k,v)
    out=attention_tiled.forward(q,k,v)

    torch.testing_assert_close(out,ref,atol=1e-3,rtol=1e-3)