from src.data import load_scifact
from src.reranker import Reranker
from src.pipeline import run_query

def test_determinism():
    docs,queries,qrels=load_scifact()
    reranker=Reranker(device="cuda")
    qid=list(queries.keys())[0]
    r1=run_query(qid,docs,queries,qrels,reranker)
    r2=run_query(qid,docs,queries,qrels,reranker)
    assert [x[0] for x in r1] == [x[0] for x in r2]

def test_hand_checked_reranking():
    docs, queries, qrels = load_scifact()
    reranker = Reranker(device="cuda")
    qid = list(queries.keys())[0]

    relevant_doc_ids = list(qrels[qid])
    print("\nKNOWN RELEVANT DOCS:")
    for doc_id in relevant_doc_ids:
        print(doc_id, docs[doc_id])

    results = run_query(qid, docs, queries, qrels, reranker)

    print("\nTOP 3 RERANKED DOCS:")
    for rank, (doc_id, text, score) in enumerate(results[:3], start=1):
        print(f"{rank}. doc={doc_id}, score={score}")
        print(text)

    top_10_doc_ids = [doc_id for doc_id, _, _ in results[:10]]
    assert any(
        doc_id in top_10_doc_ids
        for doc_id in relevant_doc_ids
    ), (
        f"Known relevant doc(s) {relevant_doc_ids} "
        f"did not appear in the top 10. "
        f"Top results: {top_10_doc_ids}"
    )