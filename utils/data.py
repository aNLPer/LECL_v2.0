import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random


def dataLoader(seq_1, seq_2, seq_3, label, id2acc, batch_size):
    num_examples = len(seq_1)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size,
    num_examples)])  # 最后⼀一次可能不不⾜足⼀一个batch
    yield features.index_select(0, j), label.index_select(0,j)