# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# s = ["2012年2月3日16时许，被告人杨某驾车行驶至昌邑市某某", "2012年2月3日16时许，"]
#
# batch_enc = tokenizer.batch_encode_plus(s,
#                             add_special_tokens=False,
#                             truncation=True,
#                             padding=True,
#                             max_length=512,
#                             return_attention_mask = True,
#                             return_tensors = 'pt')
# print(batch_enc)
# print(batch_enc["input_ids"].shape)

# enc_dict = tokenizer.batch_encode_plus(s,
#                                     add_special_tokens=False,
#                                     max_length=512,
#                                     truncation=True,
#                                     padding=True,
#                                     return_attention_mask=True,
#                                     return_tensors='pt')

a = torch.tensor([[1,2,3]],dtype=torch.float32)
a.requires_grad_(True)
zero = torch.zeros_like(a)+0.00001
b = a.where(a<=2,zero)
print(a)
print(b)
