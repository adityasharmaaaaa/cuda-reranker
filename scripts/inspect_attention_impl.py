from src.reranker import Reranker

reranker=Reranker(device="cuda")
model=reranker.model.model

print("model class:", type(model))
print("config._attn_implementation:", getattr(model.config, "_attn_implementation", "not set"))

for name, module in model.named_modules():
    if "attention" in name.lower() and "self" in name.lower():
        print(f"\nfirst self-attention module found: {name}")
        print("class:", type(module))
        print(module)
        break