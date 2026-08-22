import statistics
from src.data import load_scifact
from src.reranker import Reranker
from src.pipeline import run_query

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")

qids = [qid for qid in queries if qid in qrels and len(qrels[qid]) > 0][:20]

ranks = []
top10_hits = 0
for qid in qids:
    relevant_doc_ids = list(qrels[qid])
    results = run_query(qid, docs, queries, qrels, reranker)
    all_ranked_ids = [doc_id for doc_id, _, _ in results]
    best_rank = min(
        (all_ranked_ids.index(rid) + 1 for rid in relevant_doc_ids if rid in all_ranked_ids),
        default=None,
    )
    if best_rank is not None:
        ranks.append(best_rank)
        if best_rank <= 10:
            top10_hits += 1
    print(f"{qid}: {queries[qid][:70]!r} -> best relevant rank {best_rank} of 50")

print(f"\nqueries evaluated: {len(ranks)}")
print(f"median rank: {statistics.median(ranks)}")
print(f"top-10 hit rate: {top10_hits}/{len(ranks)}")