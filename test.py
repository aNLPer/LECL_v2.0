# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel
#
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# s = [["2012年2月3日16时许，被告人杨某驾车行驶至昌邑市某某", "2012年2月3日16时许，"],
#      ["2012年2月3日16", "2012年2月3日16时许，"]]
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
# print(enc_dict)

loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()