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

tokenizer = reranker.model.tokenizer
model = reranker.model.model
model.eval()

queries_batch = [queries[qid]] * len(candidate_texts)

def run_once():
    with record_function("tokenize"):
        features = tokenizer(
            queries_batch, candidate_texts,
            padding=True, truncation=True, return_tensors="pt",
        )
        features = {k: v.to("cuda") for k, v in features.items()}
    with record_function("forward"):
        with torch.no_grad():
            model(**features)
    torch.cuda.synchronize()

for _ in range(5):
    run_once()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    run_once()

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))