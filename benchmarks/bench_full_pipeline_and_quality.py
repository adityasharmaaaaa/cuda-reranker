import statistics
import time
import torch
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker
from src.pipeline import run_query
from src.custom_attention import register_custom_attention
import json, os
os.makedirs("results", exist_ok=True)
register_custom_attention()

docs, queries, qrels = load_scifact()
reranker_baseline = Reranker(device="cuda")
reranker_custom = Reranker(device="cuda")
reranker_custom.model.model.config._attn_implementation = "custom_cuda"

qids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0][:20]

def timed_rerank(reranker, qid):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = run_query(qid, docs, queries, qrels, reranker)
    torch.cuda.synchronize()
    return result, (time.perf_counter() - t0) * 1000

baseline_times, custom_times = [], []
exact_top10_matches = 0

for qid in qids:
    base_result, base_t = timed_rerank(reranker_baseline, qid)
    cust_result, cust_t = timed_rerank(reranker_custom, qid)
    baseline_times.append(base_t)
    custom_times.append(cust_t)

    base_top10 = [doc_id for doc_id, _, _ in base_result[:10]]
    cust_top10 = [doc_id for doc_id, _, _ in cust_result[:10]]
    if base_top10 == cust_top10:
        exact_top10_matches += 1
    else:
        overlap = len(set(base_top10) & set(cust_top10))
        print(f"{qid}: top-10 order differs (overlap {overlap}/10)")
result = {
    "queries_evaluated": len(qids),
    "exact_top10_matches": exact_top10_matches,
    "baseline_median_ms": statistics.median(baseline_times),
    "custom_median_ms": statistics.median(custom_times),
    "speedup": statistics.median(baseline_times) / statistics.median(custom_times),
}
print(result)
with open("results/stage7_full_pipeline.json", "w") as f:
    json.dump(result, f, indent=2)