from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L6-v2",device="cuda"):
        self.model=CrossEncoder(model_name,device=device)

    def rerank(self,query,candidate_texts,candidate_ids):
        pairs=[(query,text) for text in candidate_texts]
        scores=self.model.predict(pairs)
        ranked=sorted(zip(candidate_ids,candidate_texts,scores),key=lambda x:x[2],reverse=True)
        return ranked

    