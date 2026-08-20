import ir_datasets
import random

def load_scifact(split="test"):
    dataset=ir_datasets.load(f"beir/scifact/{split}")
    docs={doc.doc_id: doc.text for doc in dataset.docs_iter()}
    queries={q.query_id: q.text for q in dataset.queries_iter()}
    qrels={}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, set()).add(qrel.doc_id)
    return docs,queries,qrels

def build_candidate_pool(query_id,docs,qrels,pool_size=50,seed=0):
    rng=random.Random(seed)
    relevant=list(qrels.get(query_id,set()))
    all_doc_ids=list(docs.keys())
    pool=set(relevant)
    while len(pool)<pool_size:
        pool.add(rng.choice(all_doc_ids))
    return list(pool)