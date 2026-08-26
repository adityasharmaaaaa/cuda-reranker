import json, os
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker
from benchmarks.harness import cuda_time

os.makedirs("results", exist_ok=True)

attention_naive = load(name="attention_naive", sources=["cuda/attention_naive.cu"])
attention_tiled = load(name="attention_tiled", sources=["cuda/attention_tiled.cu"])
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

num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
batch = len(candidate_texts)

print(f"real shape -> batch={batch} heads={num_heads} seq={seq_len} head_dim={head_dim}")

torch.manual_seed(0)
q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")

def run_naive(): attention_naive.forward(q, k, v)
def run_tiled(): attention_tiled.forward(q, k, v)
def run_vectorized(): attention_vectorized.forward(q,k,v)
def run_sdpa():  F.scaled_dot_product_attention(q, k, v)

naive_stats = cuda_time(run_naive, warmup=10, reps=30)
tiled_stats = cuda_time(run_tiled, warmup=10, reps=30)
vectorized_stats = cuda_time(run_vectorized,warmup=10, reps=30)
sdpa_stats  = cuda_time(run_sdpa,  warmup=10, reps=30)

print("naive kernel:", naive_stats)
print("tiled kernel:", tiled_stats)
print("vectorized_kernel:", vectorized_stats)
print("SDPA baseline:", sdpa_stats)

with open("results/stage4_tiled_bench.json", "w") as f:
    json.dump({"naive": naive_stats, "tiled": tiled_stats, "sdpa": sdpa_stats, "vectorized": vectorized_stats,
               "shape": {"batch": batch, "heads": num_heads, "seq": seq_len, "head_dim": head_dim}},
              f, indent=2)