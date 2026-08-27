import inspect
from transformers.models.bert import modeling_bert

print(inspect.getsource(modeling_bert.BertModel.forward))