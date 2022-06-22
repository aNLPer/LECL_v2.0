# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig

config = BertConfig(num_hidden_layers=6)


bert = BertModel(config).from_pretrained("bert-base-chinese")
print(bert)