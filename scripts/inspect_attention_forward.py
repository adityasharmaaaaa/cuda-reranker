import inspect
from transformers.models.bert.modeling_bert import BertSelfAttention

print(inspect.getsource(BertSelfAttention.forward))