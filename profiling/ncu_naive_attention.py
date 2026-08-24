import torch
from torch.utils.cpp_extension import load
from src.data import load_scifact, build_candidate_pool
from src.reranker import Reranker

attention_naive = load(name="attention_naive", sources=["cuda/attention_naive.cu"])

docs, queries, qrels = load_scifact()
reranker = Reranker(device="cuda")
qid = "3"
candidate_ids = build_candidate_pool(qid, docs, qrels, 50)
candidate_texts = [docs[c] for c in candidate_ids]
tokenizer = reranker.model.tokenizer
model = reranker.model.model
queries_batch = [queries[qid]] * len(candidate_texts)
features = tokenizer(queries_batch, candidate_texts, padding=True, truncation=True, return_tensors="pt")
seq_len = features["input_ids"].shape[1]
num_heads = model.config.num_attention_heads
head_dim = model.config.hidden_size // num_heads
batch = len(candidate_texts)

torch.manual_seed(0)
q = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
k = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")
v = torch.randn(batch, num_heads, seq_len, head_dim, device="cuda")

attention_naive.forward(q, k, v)  # warmup
torch.cuda.synchronize()
attention_naive.forward(q, k, v)  # the call ncu will attach to
torch.cuda.synchronize()