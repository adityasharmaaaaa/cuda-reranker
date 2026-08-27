import torch
from torch.utils.cpp_extension import load
from src.reranker import Reranker
from src.data import load_scifact, build_candidate_pool
from transformers.models.bert import modeling_bert

# 1. inspect the registry before assuming its API
registry = modeling_bert.ALL_ATTENTION_FUNCTIONS
print("registry type:", type(registry))
print("public methods:", [m for m in dir(registry) if not m.startswith("_")])

# 2. load kernel + define the conforming translation function
attention_vectorized = load(name="attention_vectorized", sources=["cuda/attention_vectorized.cu"])

def custom_attention_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, position_bias=None, **kwargs):
    if dropout not in (0.0, None):
        raise NotImplementedError("custom kernel: dropout not supported")
    # attention_mask: [batch, 1, seq, seq] bool, confirmed uniform across query rows —
    # row 0 per batch item gives the real per-candidate key count
    valid_lengths = attention_mask[:, 0, 0, :].sum(dim=-1).to(torch.int32)
    attn_output = attention_vectorized.forward(
        query.contiguous(), key.contiguous(), value.contiguous(), valid_lengths
    )
    attn_output = attn_output.transpose(1, 2).contiguous()  # match sdpa_attention_forward's output layout
    return attn_output, None

# 3. register — dynamic lookup to avoid version-specific import errors
from transformers import AttentionInterface
AttentionInterface.register("custom_cuda", custom_attention_forward)

import importlib
mask_module_path = modeling_bert.create_bidirectional_mask.__module__
print("create_bidirectional_mask defined in:", mask_module_path)
mask_module = importlib.import_module(mask_module_path)

mask_registry = mask_module.ALL_MASK_ATTENTION_FUNCTIONS
print("mask registry type:", type(mask_registry))
print("mask registry public methods:", [m for m in dir(mask_registry) if not m.startswith("_")])
print("keys before:", list(mask_registry.keys()) if hasattr(mask_registry, "keys") else "no .keys()")

sdpa_mask_fn = mask_registry["sdpa"]
mask_registry.register("custom_cuda", sdpa_mask_fn)

print("keys after:", list(mask_registry.keys()) if hasattr(mask_registry, "keys") else "no .keys()")

# 4. one baseline model, one pointed at the custom kernel
docs, queries, qrels = load_scifact()
reranker_baseline = Reranker(device="cuda")
reranker_custom = Reranker(device="cuda")
reranker_custom.model.model.config._attn_implementation = "custom_cuda"

qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]
tokenizer = reranker_baseline.model.tokenizer
queries_batch = [queries[qid]] * len(candidate_texts)
features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
features = {k: v.to("cuda") for k, v in features.items()}

reranker_baseline.model.model.eval()
reranker_custom.model.model.eval()

# 5. compare full model output, not just the isolated kernel
with torch.no_grad():
    out_baseline = reranker_baseline.model.model(**features).logits
    out_custom = reranker_custom.model.model(**features).logits

max_err = (out_baseline - out_custom).abs().max().item()
mean_err = (out_baseline - out_custom).abs().mean().item()
print(f"\nlogit max abs error: {max_err}")
print(f"logit mean abs error: {mean_err}")