import re
import random
import math
import json
import numpy as np
import pickle
from transformers import BertTokenizer
import torch
import torch.nn.functional as F


TEMPER = 1

class Lang:
    # 语料库对象
    def __init__(self, name="corpus"):
        self.name = name
        self.word2index = {"PAD":0, "UNK":1}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"UNK"}
        # 词汇表大小
        self.n_words = 2
        self.index2accu = []
        self.accu2index = None
        self.index2art = []
        self.art2index = None

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addLabel(self, accu, article):
        if accu not in self.index2accu:
            self.index2accu.append(accu)
        if article not in self.index2art:
            self.index2art.append(article)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def update_label2index(self):
        self.accu2index = {accu: idx for idx, accu in enumerate(self.index2accu)}
        self.art2index = {art: idx for idx, art in enumerate(self.index2art)}

class ConfusionMatrix:

    def __init__(self, n_class):
        """
        混淆矩阵的每一行代表真实标签，每一列代表预测标签
        :param n_class: 类别数目
        """
        self.__mat = np.zeros((n_class, n_class))
        self.n_class = n_class
        self.n_activated_class = None
        self.class_weights = None

    def updateMat(self, preds, labels):
        """
        根据预测结果和标签更新混淆矩阵
        :param preds:
        :param labels:
        :return:
        """
        # 更新矩阵数值
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        for i in range(len(labels_flat)):
            self.__mat[labels_flat[i]][pred_flat[i]] += 1

        # 更新每个类别权重
        counts = self.__mat.sum(axis=1)
        self.class_weights = counts/np.sum(counts)

        # 计算有效类别
        self.n_activated_class = sum(counts != 0)


    def get_acc(self):
        """
        :return: 返回准确率
        """
        return self.__mat.trace()/self.__mat.sum()

    def get_recall(self, class_idx):
        """
        返回某类别的召回率
        :param class_idx:
        :return:
        """
        return self.__nomal1()[class_idx][class_idx]

    def get_precision(self, class_idx):
        """
        返回某类别的精确率
        :param class_idx:
        :return:
        """
        return self.__nomal1(dim=0)[class_idx][class_idx]

    def get_f1(self, class_idx):
        """
        返回某类别的f1值
        :param class_idx:
        :return:
        """
        recall = self.get_recall(class_idx)
        precision = self.get_precision(class_idx)
        return 2*recall*precision / (recall+precision+0.00001)

    def getMaP(self):
        """
        :return: 返回 Macro-Precision
        """
        norm_mat = self.__nomal1(dim=0)
        return norm_mat.trace()/self.n_activated_class

    def getMiP(self):
        """
        返回 Micro-Precision, 考虑到每个类别的权重适合分布不均衡数据
        :return:
        """
        pass

    def getMaR(self):
        """
        :return: 返回 Macro-Recall
        """
        norm_mat = self.__nomal1(dim=1)
        return norm_mat.trace() / self.n_activated_class

    def getMiR(self):
        """
        返回 Micro-Recall, 考虑到每个类别的权重适合分布不均衡数据
        :return:
        """
        pass

    def getMaF(self):
        """
        返回 Macro-F1
        :return:
        """
        mat_1 = self.__nomal1()
        mat_0 = self.__nomal1(dim=0)
        mat = 2*mat_0*mat_1/(mat_1+mat_0+0.00001)
        return mat.trace()/self.n_activated_class


    def getMiF(self):
        pass

    def __nomal1(self, dim=1):
        """
        归一化矩阵
        :param dim: 指定在哪个维度做归一化 default：1
        :return:
        """
        assert dim == 1 or dim == 0
        if dim == 1:
            return self.__mat / (self.__mat.sum(axis=dim)[:, np.newaxis]+0.00001)
        else:
            return self.__mat / ((self.__mat.sum(axis=dim)+0.000001))


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
def filter_dict(data_dict, bound=100):
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
                         lang,
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
    accu_labels = []
    article_labels = []
    penalty_labels = []
    seq = []
    for _ in range(positive_size):
        seq.append([])
        accu_labels.append([])
        article_labels.append([])
        penalty_labels.append([])
    # 获取数据集中的所有指控
    accus = np.array(list(accu2case.keys()))
    # 选取指控
    sample_accus = list(np.random.choice(accus, size=int(batch_size/(positive_size*sim_accu_num)), replace=False))
    selected_accus = sample_accus.copy()
    count = 0
    while count<sim_accu_num-1:
        for accu in sample_accus:
            # 获取相似指控
            accu = lang.index2accu[accu]
            sim_accu_ = [category2accu[c] for c in accu2category[accu]]
            temp = []
            for l in sim_accu_:
                # 筛选出在数据集中出现的相似指控
                for i in l:
                    if i in lang.accu2index:
                        i = lang.accu2index[i]
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
            seq[i].append(selected_cases[i][0])
            accu_labels[i].append(selected_cases[i][1])
            article_labels[i].append(selected_cases[i][2])
            penalty_labels[i].append(selected_cases[i][3])

    return seq, accu_labels, article_labels, penalty_labels





def data_loader(seq, label, batch_size):
    num_examples = len(seq)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        ids = indices[i: min(i + batch_size, num_examples)]  # 最后⼀次可能不⾜⼀个batch
        yield [seq[j] for j in ids], [label[j] for j in ids]

def train_distloss_fun(outputs, radius = 10):
    """
    :param outputs: [posi_size, batch_size/posi_size, hidden_dim]
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

def accumulated_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), len(labels_flat)

def genConfusMat(confusMat, preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    for i in range(len(labels_flat)):
        confusMat[labels_flat[i]][pred_flat[i]] += 1

def prepare_valid_data(resourcefile):
    seq = []
    label = []
    f = open("../dataprepare/lang_data_train_preprocessed.pkl", "rb")
    lang = pickle.load(f)
    f.close()
    with open(resourcefile, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            seq.append(example[0])
            label.append(lang.label2index[example[1]])
    return seq, label



