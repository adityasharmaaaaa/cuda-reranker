import json
from benchmarks.harness import cuda_time
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker

docs, queries, qrels=load_scifact()
reranker=Reranker(device="cuda")
qid=list(queries.keys())[0]
candidate_ids=build_candidate_pool(qid,docs,qrels,50)
candidate_texts=[docs[c] for c in candidate_ids]

def fn():
    reranker.rerank(queries[qid],candidate_texts,candidate_ids)

stats=cuda_time(fn,warmup=5,reps=20)
print(stats)
with open("results/stage1_baseline.json","w") as f:
    json.dump(stats,f,indent=2)