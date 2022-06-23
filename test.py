# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig
from collections import OrderedDict

d = OrderedDict.fromkeys('acbde')
print(d["d"])