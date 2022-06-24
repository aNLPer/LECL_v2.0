# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig
from collections import OrderedDict
from timeit import default_timer as timer
import time

print("hello python")

start = timer()
time.sleep(5)
end = timer()
print(end-start)


