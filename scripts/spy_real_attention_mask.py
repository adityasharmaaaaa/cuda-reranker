import torch
import torch.nn.functional as F
from src.reranker import Reranker
from src.data import load_scifact, build_candidate_pool

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
model = reranker.model.model
tokenizer = reranker.model.tokenizer

qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]
queries_batch = [queries[qid]] * len(candidate_texts)
features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
features = {k: v.to("cuda") for k, v in features.items()}

original_sdpa = F.scaled_dot_product_attention
captured = {}

def spy_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if "seen" not in captured:
        captured["seen"] = True
        captured["query_shape"] = query.shape
        captured["attn_mask"] = attn_mask
        captured["dropout_p"] = dropout_p
        captured["is_causal"] = is_causal
        captured["scale"] = scale
    return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

F.scaled_dot_product_attention = spy_sdpa
model.eval()
with torch.no_grad():
    model(**features)
F.scaled_dot_product_attention = original_sdpa  # restore immediately, before anything else runs

print("query shape:", captured["query_shape"])
print("scale:", captured["scale"], " (expected 1/sqrt(32) =", 32**-0.5, ")")
print("is_causal:", captured["is_causal"])
print("dropout_p:", captured["dropout_p"])

mask = captured["attn_mask"]
if mask is None:
    print("attn_mask is None")
else:
    print("attn_mask shape:", mask.shape, " dtype:", mask.dtype)
    print("unique values (first 10):", torch.unique(mask)[:10])
    print("row 0, first 20 entries:", mask.flatten(0, -2)[0, :20] if mask.dim() > 1 else mask[:20])