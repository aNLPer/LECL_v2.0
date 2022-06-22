import re
import math
import json
import numpy as np
from transformers import BertTokenizer
import torch
import torch.nn.functional as F

TEMPER = 1

class Lang:
    # 语料库对象
    def __init__(self, name):
        self.name = name
        self.word2index = {"UNK":0, "SOS":1, "EOS":3}
        self.word2count = {}
        self.index2word = {0:"UNK", 1:"SOS", 2:"EOS"}
        # 词汇表大小
        self.n_words = 3
        self.index2label = []
        self.label2index = None

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addLabel(self, label):
        if label not in self.index2label:
            self.index2label.append(label)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def update_label2index(self):
        self.label2index = {accu:idx for idx, accu in enumerate(self.index2label)}

def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

# 大写数字转阿拉伯数字
def hanzi_to_num(hanzi_1):
    # for num<10000
    hanzi = hanzi_1.strip().replace('零', '')
    if hanzi == '':
        return str(int(0))
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '': 0}
    m = {'十': 1e1, '百': 1e2, '千': 1e3, }
    w = {'万': 1e4, '亿': 1e8}
    res = 0
    tmp = 0
    thou = 0
    for i in hanzi:
        if i not in d.keys() and i not in m.keys() and i not in w.keys():
            return hanzi

    if (hanzi[0]) == '十': hanzi = '一' + hanzi
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp += d[hanzi[i]]
        elif hanzi[i] in m:
            tmp *= m[hanzi[i]]
            res += tmp
            tmp = 0
        else:
            thou += (res + tmp) * w[hanzi[i]]
            tmp = 0
            res = 0
    return int(thou + res + tmp)

# 过滤掉值小于100的项目
def filter_dict(data_dict, bound):
    return {k: v for k, v in data_dict.items() if v >= bound}

# 对字典中的每个项目求和
def sum_dict(data_dict):
    sum = 0
    for k,v in data_dict.items():
        sum+= v
    return sum

# 字典重置
def reset_dict(data_dict):
    return {k: 0 for k, v in data_dict.items()}

# 加载停用词表、特殊符号表、标点
def get_filter_symbols(filepath):
    '''
    根据mode加载标点、特殊词或者停用词
    :param mode:
    :return:list
    '''
    return list(set([line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]))

# law内容过滤
def filterStr(law):
    # 删除第一个标点之前的内容
    pattern_head_content = re.compile(r".*?[，：。,:.]")
    head_content = pattern_head_content.match(law)
    if head_content is not None:
        head_content_span = head_content.span()
        law = law[head_content_span[1]:]

    # 删除“讼诉机关认为，......”
    pattern_3 = re.compile(r"[，。]公诉机关")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]+1]

    # 删除"。...事实，"
    pattern_3 = re.compile(r"。.{2,8}事实，")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]]

    # 删除括号及括号内的内容
    pattern_bracket = re.compile(r"[<《【\[(（〔].*?[〕）)\]】》>]")
    law = pattern_bracket.sub("", law)

    return law

# 生成acc2desc字典
def get_acc_desc(file_path):
    acc2desc = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            dict = json.loads(line)
            if dict["accusation"] not in acc2desc:
                acc2desc[dict["accusation"]] = dict["desc"]
    return acc2desc


# 获取batch
def pretrain_data_loader(accu2case,
                         batch_size,
                         positive_size=2,
                         sim_accu_num=2,
                         category2accu=None,
                         accu2category=None):
    """
    1. 先从accu2case中抽取出batch_size/2种不同的指控
    2. 对于每个指控随机抽取positive_size个案件

    :param batch_size:  = positive_size * sim_accu_num * sample_accus
    :param positive_size: 正样本数量
    :param accu2case: 指控：案件s （字典）
    :param category2accu: 类别：[指控s]（字典）
    :param accu2category: 指控：[类别s]（字典）
    :param sim_accu_num: 一个batch中任意指控及其相似指控的和

    其中 batch_size/sim_accu_num 为整数
    :return:
    """
    seq = []
    for _ in range(positive_size):
        seq.append([])
    # 获取数据集中的所有指控
    accus = np.array(list(accu2case.keys()))
    # 选取指控
    sample_accus = list(np.random.choice(accus, size=int(batch_size/(positive_size*sim_accu_num)), replace=False))
    selected_accus = sample_accus.copy()
    count = 0
    while count<sim_accu_num-1:
        for accu in sample_accus:
            # 获取相似指控
            sim_accu_ = [category2accu[c] for c in accu2category[accu]]
            temp = []
            for l in sim_accu_:
                # 筛选出在数据集中出现的相似指控
                for i in l:
                    if i in accus:
                        temp.append(i)
                # temp.extend([i for i in l and i in accus])
            # 去除相似指控与selected_accus指控中的重复指控
            temp = set(temp)
            temp = temp.difference(set(selected_accus))
            # 添加不重复的相似指控
            sim_accu = list(temp)
            if len(sim_accu) != 0:
                selected_accus.extend(np.random.choice(sim_accu, size=1))
        count+=1

    # 若获取的指控不足则随机挑选补充
    if len(selected_accus) < batch_size / positive_size:
        bias = int(batch_size/positive_size-len(selected_accus))
        selected_accus.extend(np.random.choice(list(set(accus).difference(set(selected_accus))), size=bias, replace=False))
    # print(len(set(selected_accus)))
    # 根据指控获取batch
    for accu in selected_accus:
        selected_cases = np.random.choice(accu2case[accu], size=positive_size, replace=False)
        for i in range(positive_size):
            seq[i].append(selected_cases[i])

    return seq


def train_cosloss_fun(out_1, out_2, out_3, label_rep):
    """
    损失函数
    :param out_1: tensor
    :param out_2: tensor
    :param out_3: tensor
    :param label_rep tensor
    :return: loss scalar
    """
    batch_size = out_1.shape[0]
    # out_1 样本损失函数
    loss_out1 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_1[i].expand(batch_size, -1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1)/TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1)/TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1)/TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1)/TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out2[i]), torch.exp(x_out3[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1[0:i])) - torch.exp(x_out1[i]) + torch.sum(torch.exp(x_out2)) + torch.sum(torch.exp(x_out3)) + torch.sum(torch.exp(x_label_rep))
        loss_out1 -= torch.log(molecule/denominator)

    # out_2 样本损失函数
    loss_out2 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_2[i].expand(batch_size, -1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1) / TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1) / TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1) / TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1) / TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out1[i]), torch.exp(x_out3[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1)) + torch.sum(torch.exp(x_out2)) - torch.exp(x_out2[i]) + torch.sum(torch.exp(x_out3)) + torch.sum(torch.exp(x_label_rep))
        loss_out2 -= torch.log(molecule / denominator)

    # out_3 样本损失函数
    loss_out3 = 0
    for i in range(batch_size):
        # [batch_size, d_model]
        x = out_3[i].expand(batch_size, -1)
        # [batch_size]
        x_out1 = torch.cosine_similarity(x, out_1, dim=1) / TEMPER
        # [batch_size]
        x_out2 = torch.cosine_similarity(x, out_2, dim=1) / TEMPER
        # [batch_size]
        x_out3 = torch.cosine_similarity(x, out_3, dim=1) / TEMPER
        # [batch_size]
        x_label_rep = torch.cosine_similarity(x, label_rep, dim=1) / TEMPER

        molecule = torch.sum(torch.tensor([torch.exp(x_out1[i]), torch.exp(x_out2[i]), torch.exp(x_label_rep[i])]))
        denominator = torch.sum(torch.exp(x_out1)) + torch.sum(torch.exp(x_out2)) + torch.sum(torch.exp(x_out3)) - torch.exp(x_out3[i]) + torch.sum(torch.exp(x_label_rep))
        loss_out3 -= torch.log(molecule / denominator)

    return loss_out1 + loss_out2 + loss_out3

def train_distloss_fun(outputs, radius = 10):
    """
    :param outputs: [posi_size, batch_size/2, hidden_dim]
    :param label_rep:
    :param label:
    :return:
    """
    posi_size = outputs.shape[0]
    batch_size = outputs.shape[1]
    # 正样本距离
    posi_pairs_dist =0
    for i in range(posi_size-1):
        for j in range(i+1, posi_size):
            posi_pairs_dist += torch.sum(F.pairwise_distance(outputs[i], outputs[j]))

    # 负样本距离
    # [posi_size, batch_size/2, hidden_dim] -> [batch_size/2, posi_size,  hidden_dim]
    outputs = torch.transpose(outputs, dim0=0, dim1=1)
    neg_pairs_dist = 0
    for i in range(int(0.5*batch_size)-1):
        for j in range(i+1, int(0.5*batch_size)):
            # outputs[i] outputs[j]
            for k in range(posi_size):
                dist = F.pairwise_distance(outputs[i][k], outputs[j])
                zero = torch.zeros_like(dist)+0.00001
                dist = dist.where(dist<radius, zero)
                neg_pairs_dist += torch.sum(dist)

    return posi_pairs_dist/batch_size, \
           neg_pairs_dist/batch_size