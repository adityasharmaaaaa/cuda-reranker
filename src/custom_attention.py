import torch
import importlib
from torch.utils.cpp_extension import load
from transformers import AttentionInterface
from transformers.models.bert import modeling_bert

_registered = False

def register_custom_attention():
    global _registered
    if _registered:
        return
    attention_vectorized = load(name="attention_vectorized", sources=["cuda/attention_vectorized.cu"])

    def custom_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, position_bias=None, **kwargs):
        if dropout not in (0.0, None):
            raise NotImplementedError("custom kernel: dropout not supported")
        valid_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int32)
        attn_output = attention_vectorized.forward(
            query.contiguous(), key.contiguous(), value.contiguous(), valid_lengths
        )
        return attn_output.transpose(1, 2).contiguous(), None

    AttentionInterface.register("custom_cuda", custom_attention_forward)

    mask_module = importlib.import_module(modeling_bert.create_bidirectional_mask.__module__)
    mask_registry = mask_module.ALL_MASK_ATTENTION_FUNCTIONS
    mask_registry.register("custom_cuda", mask_registry["sdpa"])

    _registered = True