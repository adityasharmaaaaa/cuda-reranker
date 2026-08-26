from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]

tokenizer = reranker.model.tokenizer
queries_batch = [queries[qid]] * len(candidate_texts)
features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")

seq_len = features["input_ids"].shape[1]
real_lengths = features["attention_mask"].sum(dim=1)

print(f"batch padded to seq_len={seq_len}")
print(f"real (non-pad) lengths: min={real_lengths.min().item()}, max={real_lengths.max().item()}, mean={real_lengths.float().mean().item():.1f}")
print(f"candidates shorter than seq_len (i.e. have padding): {(real_lengths < seq_len).sum().item()} / {len(real_lengths)}")