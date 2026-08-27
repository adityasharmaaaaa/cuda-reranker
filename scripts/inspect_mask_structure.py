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
features_cuda = {k: v.to("cuda") for k, v in features.items()}

original_sdpa = F.scaled_dot_product_attention
captured = {}

def spy_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    if "seen" not in captured:
        captured["seen"] = True
        captured["attn_mask"] = attn_mask.clone()
    return original_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)

F.scaled_dot_product_attention = spy_sdpa
model.eval()
with torch.no_grad():
    model(**features_cuda)
F.scaled_dot_product_attention = original_sdpa

mask = captured["attn_mask"]  # [batch, 1, seq, seq], bool
tokenizer_valid_lengths = features["attention_mask"].sum(dim=1)
seq = mask.shape[2]

mask_valid_from_row0 = mask[:, 0, 0, :].sum(dim=-1).cpu()
print("tokenizer-derived valid lengths (first 10):", tokenizer_valid_lengths[:10].tolist())
print("mask-derived valid lengths, from query row 0 (first 10):", mask_valid_from_row0[:10].tolist())
print("match:", torch.equal(tokenizer_valid_lengths, mask_valid_from_row0))

example_batch_idx = (tokenizer_valid_lengths < seq).nonzero()[0].item()
real_len = tokenizer_valid_lengths[example_batch_idx].item()
print(f"\nbatch item {example_batch_idx}, real length {real_len}")

row_q0 = mask[example_batch_idx, 0, 0, :]
row_q_mid = mask[example_batch_idx, 0, real_len // 2, :]
print("mask row at query 0 == mask row at query", real_len // 2, ":", torch.equal(row_q0, row_q_mid))

row_pad_query = mask[example_batch_idx, 0, real_len, :]  # first padding query position
print("padding-query row: all False?", (row_pad_query == False).all().item(),
      " all True?", (row_pad_query == True).all().item(),
      " sum:", row_pad_query.sum().item())