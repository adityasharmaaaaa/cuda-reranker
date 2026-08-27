import json, os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker
from benchmarks.harness import cuda_time

os.makedirs("results", exist_ok=True)

attention_vectorized = load(name="attention_vectorized", sources=["cuda/attention_vectorized.cu"])

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]

tokenizer = reranker.model.tokenizer
model = reranker.model.model
queries_batch = [queries[qid]] * len(candidate_texts)
features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
seq_len = features["input_ids"].shape[1]
valid_lengths = features["attention_mask"].sum(dim=1).to(torch.int32)

num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
batch = len(candidate_texts)

print(f"real shape -> batch={batch} heads={num_heads} seq={seq_len} head_dim={head_dim}")
print(f"valid lengths: min={valid_lengths.min().item()} max={valid_lengths.max().item()} mean={valid_lengths.float().mean().item():.1f}")

torch.manual_seed(0)
q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
valid_lengths_cuda = valid_lengths.to("cuda")

key_idx = torch.arange(seq_len)
bool_mask = key_idx.unsqueeze(0) < valid_lengths.unsqueeze(1)
attn_mask = bool_mask[:, None, None, :].to(device="cuda")

def run_vectorized_masked(): attention_vectorized.forward(q, k, v, valid_lengths_cuda)
def run_sdpa_masked():       F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
def run_sdpa_unmasked():     F.scaled_dot_product_attention(q, k, v)  # kept for reference only — not the real comparison

vec_stats  = cuda_time(run_vectorized_masked, warmup=10, reps=30)
sdpa_stats = cuda_time(run_sdpa_masked, warmup=10, reps=30)
sdpa_unmasked_stats = cuda_time(run_sdpa_unmasked, warmup=10, reps=30)

print("vectorized (masked):", vec_stats)
print("SDPA (masked):", sdpa_stats)
print("SDPA (unmasked, reference only):", sdpa_unmasked_stats)

with open("results/stage4_masked_bench.json", "w") as f:
    json.dump({"vectorized_masked": vec_stats, "sdpa_masked": sdpa_stats, "sdpa_unmasked": sdpa_unmasked_stats,
               "valid_lengths": {"min": valid_lengths.min().item(), "max": valid_lengths.max().item(), "mean": valid_lengths.float().mean().item()},
               "shape": {"batch": batch, "heads": num_heads, "seq": seq_len, "head_dim": head_dim}}, f, indent=2)