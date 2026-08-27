import inspect
from transformers.models.bert import modeling_bert

sdpa_fn = modeling_bert.ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", None)
print("resolved function:", sdpa_fn)
print()
print(inspect.getsource(sdpa_fn))