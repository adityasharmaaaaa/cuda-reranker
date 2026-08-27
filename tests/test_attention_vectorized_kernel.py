import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

attention_vectorized = load(name="attention_vectorized", sources=["cuda/attention_vectorized.cu"], verbose=True)

def test_attention_vectorized_matches_reference():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 16, 32
    q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v)
    out = attention_vectorized.forward(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

def test_attention_vectorized_partial_tile():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 2, 4, 48, 32  # 48 = one full 32-tile + a 16-row partial tile
    q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    ref = F.scaled_dot_product_attention(q, k, v)
    out = attention_vectorized.forward(q, k, v)
    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)

def test_attention_vectorized_masked():
    torch.manual_seed(0)
    batch, heads, seq, head_dim = 4, 4, 128, 32
    q = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    k = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)
    v = torch.randn(batch, heads, seq, head_dim, device="cuda", dtype=torch.float32)

    valid_lengths = torch.tensor([30, 128, 45, 96], dtype=torch.int32)  # deliberately includes a short one, near tile boundaries, and one fully unpadded

    key_idx = torch.arange(seq)
    bool_mask = key_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)          # [batch, seq], True = real token
    attn_mask = bool_mask[:, None, None, :].to(device="cuda")              # broadcasts over heads and query positions

    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = attention_vectorized.forward(q, k, v, valid_lengths.to("cuda"))

    torch.testing.assert_close(out, ref, atol=1e-3, rtol=1e-3)