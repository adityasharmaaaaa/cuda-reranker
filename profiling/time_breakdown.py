import time
import torch
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]

tokenizer = reranker.model.tokenizer
model = reranker.model.model
model.eval()

print("tokenizer.is_fast:", tokenizer.is_fast)  # rust-backed fast tokenizer, or slow python path?

queries_batch = [queries[qid]] * len(candidate_texts)

def time_tokenize(reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
        times.append((time.perf_counter() - t0) * 1000)
    return times

def time_forward(reps):
    features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
    features = {k: v.to("cuda") for k, v in features.items()}
    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(**features)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return times

time_tokenize(5); time_forward(5)  # warmup, untimed

tok = sorted(time_tokenize(20))
fwd = sorted(time_forward(20))
print(f"tokenize ms: median {tok[len(tok)//2]:.2f}  min {tok[0]:.2f}  max {tok[-1]:.2f}")
print(f"forward  ms: median {fwd[len(fwd)//2]:.2f}  min {fwd[0]:.2f}  max {fwd[-1]:.2f}")