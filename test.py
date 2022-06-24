# coding=utf-8
import transformers
import torch
from torch.nn import PairwiseDistance
import torch.nn.functional as F
import torch.nn as nn
from utils.commonUtils import ConfusionMatrix
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertConfig
from collections import OrderedDict
from timeit import default_timer as timer
import time
import numpy as np
for _ in range(3):
    confusMat = ConfusionMatrix(3)
    preds = np.array([[0.1, 0.5,0.4],[0.5, 0.3,0.2],[0.3,0.3,0.4],[0.1,0.5,0.4],[0.4,0.5,0.1]])
    label = np.array([1,2,0,1,0])
    confusMat.updateMat(preds, label)
    print(confusMat.get_acc())
    print(confusMat.get_precision(1))
    print(confusMat.get_recall(1))
    print(confusMat.get_f1(1))
    print(confusMat.getMaF())
    print((confusMat.get_f1(0)+confusMat.get_f1(1)+confusMat.get_f1(2))/3)
    print(confusMat.getMaP())
    print((confusMat.get_precision(0)+confusMat.get_precision(1)+confusMat.get_precision(2))/3)
    print(confusMat.getMaR())
    print((confusMat.get_recall(0)+confusMat.get_recall(1)+confusMat.get_recall(2))/3)
    print("end")


def genConfusMat(cm,i,j):
    cm[i][j] += i*j

for i in range(3):
    for j in range(3):
        genConfusMat(confusMat, i, j)
print(confusMat)
print(2*confusMat*confusMat/(confusMat+confusMat+0.0001))
confusMat[1][1] = 2
confusMat[1][2] = 3
print(confusMat)
counts = confusMat.sum(axis=1)
print(counts)
print(counts/np.sum(counts))
