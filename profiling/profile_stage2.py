import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker

os.makedirs("profiling", exist_ok=True) 

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]

for _ in range(5):  # warmup
    reranker.rerank(queries[qid], candidate_texts, candidate_ids)
torch.cuda.synchronize()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    with record_function("rerank_full"):
        reranker.rerank(queries[qid], candidate_texts, candidate_ids)
    torch.cuda.synchronize()

print("=== sorted by CUDA time ===")
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
print("\n=== sorted by CPU time ===")
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))

prof.export_chrome_trace("profiling/stage2_trace.json")