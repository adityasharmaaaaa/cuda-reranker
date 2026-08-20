from src.data import build_candidate_pool

def run_query(query_id,docs,queries,qrels,reranker,pool_size=50):
    query_text=queries[query_id]
    candidate_ids=build_candidate_pool(query_id,docs,qrels,pool_size)
    candidate_texts=[docs[cid] for cid in candidate_ids]
    return reranker.rerank(query_text,candidate_texts,candidate_ids)