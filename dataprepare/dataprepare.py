# coding:utf-8
import configparser
import os
import pickle
import re
import json
import thulac
import utils.commonUtils as commonUtils
from utils.commonUtils import Lang
from utils import Law
from transformers import BertModel, BertTokenizer
import numpy as np
import operator

config = configparser.ConfigParser()
print("read project config......")
config.read("../config.cfg")
file_names = ["train", "test"]
folders = ["CAIL-SMALL", "CAIL-LARGE"]
data_base_path = config.get("dataprepare", "data_base_path")


def data_filter():
    """
    根据文本长度、样本频率过滤数据集
    :param source_path:
    :param target_path:
    :return:
    """
    for folder in folders:
        dict_articles = {}  # 法律条款：数量
        dict_accusations = {}  # 指控：数量
        for i in range(len(file_names)):
            print(f'{folder}/{file_names[i]}.json statistic beginning')
            with open(os.path.join(data_base_path, folder, f"{file_names[i]}.json"), 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    example_articles = example['meta']['relevant_articles']
                    example_accusation = example['meta']['accusation']
                    example_fact = example['fact']
                    # 仅统计单条款、单指控、仅一审的案件的指控和条款
                    if len(example_articles) == 1 and \
                            len(example_accusation) == 1 and \
                            '二审' not in example_fact and \
                            len(example_fact) > 10:
                        if dict_articles.__contains__(example_articles[0]):
                            dict_articles[example_articles[0]] += 1
                        else:
                            dict_articles.update({example_articles[0]: 1})
                        if dict_accusations.__contains__(example_accusation[0]):
                            dict_accusations[example_accusation[0]] += 1
                        else:
                            dict_accusations.update({example_accusation[0]: 1})
            print(f'{folder}/{file_names[i]}.json statistic over')

        # 过滤掉频次小于100的条款和指控
        dict_articles = commonUtils.filter_dict(dict_articles)
        dict_accusations = commonUtils.filter_dict(dict_accusations)

        articles_sum = commonUtils.sum_dict(dict_articles)
        accusation_sum = commonUtils.sum_dict(dict_accusations)

        print('filter begining......')
        while articles_sum != accusation_sum:
            dict_accusations = commonUtils.reset_dict(dict_accusations)
            dict_articles = commonUtils.reset_dict(dict_articles)
            for i in range(len(file_names)):
                with open(os.path.join(data_base_path, folder, f"{file_names[i]}.json"), 'r', encoding='utf-8') as f:
                    for line in f:
                        example = json.loads(line)
                        example_articles = example['meta']['relevant_articles']
                        example_accusation = example['meta']['accusation']
                        example_fact = example['fact']
                        if len(example_articles) == 1 and \
                                len(example_accusation) == 1 and \
                                '二审' not in example_fact:
                            # 该案件对应的article和accusation频率都大于100
                            if dict_articles.__contains__(example_articles[0]) and \
                                    dict_accusations.__contains__(example_accusation[0]) and \
                                    len(example_fact) > 10:
                                dict_articles[example_articles[0]] += 1
                                dict_accusations[example_accusation[0]] += 1
                            else:
                                continue
            dict_articles = commonUtils.filter_dict(dict_articles)
            dict_accusations = commonUtils.filter_dict(dict_accusations)

            articles_sum = commonUtils.sum_dict(dict_articles)
            accusation_sum = commonUtils.sum_dict(dict_accusations)

            print('articles_num: ' + str(len(dict_articles)))
            print('article_sum: ' + str(articles_sum))

            print('accusation_num=' + str(len(dict_accusations)))
            print('accusation_sum: ' + str(accusation_sum))
            print("\n\n")

        for i in range(len(file_names)):
            f1 = open(os.path.join(data_base_path, folder, f"{file_names[i]}_filtered.json"), "w", encoding="utf-8")
            with open(os.path.join(data_base_path, folder, f"{file_names[i]}.json"), 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    example_articles = example['meta']['relevant_articles']
                    example_accusation = example['meta']['accusation']
                    example_fact = example['fact']
                    if len(example_articles) == 1 and \
                            len(example_accusation) == 1 and \
                            '二审' not in example_fact and \
                            len(example_fact) >= 10:
                        # 该案件对应的article和accusation频率都大于100
                        if dict_articles.__contains__(example_articles[0]) and \
                                dict_accusations.__contains__(example_accusation[0]) and \
                                len(example_fact) > 10:
                            f1.write(line)
                        else:
                            continue
                f1.close()
        print('filter over......')

# 构造数据集
def data_process():
    '''
    构造数据集：[case_desc, "acc", "article","penalty"]
    # 分词
    # 去除特殊符号
    # 去除停用词
    # 去除标点
    # 去除停用词和标点
    # 同类别的accusation(样本+4)
    :param case_path: 案件描述文件
    :param acc2desc: 指控：指控描述 （字典）
    :return: [[[case_desc,case_desc,...], "acc", "acc_desc"],]
    '''

    # 加载分词器
    # thu = thulac.thulac(user_dict="Thuocl_seg.txt", seg_only=True)
    # 加载停用词表
    # stopwords = commonUtils.get_filter_symbols("stop_word.txt")
    # 加载标点
    # punctuations = commonUtils.get_filter_symbols("punctuation.txt")
    # 加载特殊符号
    special_symbols = commonUtils.get_filter_symbols("special_symbol.txt")
    for folder in folders:
        lang = Lang(folder)
        for file in file_names:
            print(f"start processing data in folder {folder}/{file}_filtered.json")
            fw = open(os.path.join(data_base_path, folder, f"{file}_processed.txt"), "w", encoding="utf-8")
            count = 0
            with open(os.path.join(data_base_path, folder, f"{file}_filtered.json"), "r", encoding="utf-8") as f:
                for line in f:
                    count += 1
                    item = [] # 单条训练数据
                    example = json.loads(line)
                    # 过滤law article内容
                    example_fact = Law.filterStr(example["fact"])
                    example_penalty = example["meta"]["term_of_imprisonment"]
                    # 去除特殊符号
                    example_fact_1 = [char for char in example_fact if char not in special_symbols]
                    lang.addLabel(example["meta"]['accusation'][0], example["meta"]['relevant_articles'][0])
                    lang.update_label2index()
                    # example_fact_1 = [word for word in thu.cut(example_fact, text=True).split(" ") if word not in special_symbols]
                    #example_fact_1 = [re.sub(r"\d+", "num", word) for word in example_fact_1]
                    #example_fact_1 = [word for word in example_fact_1
                    #                  if word not in ["x年", "x月", "x日", "下午", "上午", "凌晨", "晚", "晚上", "x时", "x分", "许"]]
                    # 去除停用词
                    # example_fact_2 = [word for word in example_fact_1 if word not in stopwords]
                    # 去除标点
                    # example_fact_3 = [word for word in example_fact_1 if word not in punctuations and word not in stopwords]
                    # 去除停用词和标点
                    #example_fact_4 = [word for word in example_fact_1 if word not in punctuations and word not in stopwords]
                    # facts = [example_fact_3, example_fact_4]
                    # item.append(example_fact_2)
                    # item.append(example_fact_3)
                    if len(example_fact_1)<10:
                        continue
                    item.append("".join(example_fact_1).strip())
                    item.append(lang.accu2index[example['meta']['accusation'][0]])
                    item.append(lang.art2index[example['meta']['relevant_articles'][0]])
                    if (example_penalty["death_penalty"] == True or example_penalty["life_imprisonment"] == True):
                        item.append(0)
                    elif example_penalty["imprisonment"] > 10 * 12:
                         item.append(1)
                    elif example_penalty["imprisonment"] > 7 * 12:
                         item.append(2)
                    elif example_penalty["imprisonment"] > 5 * 12:
                         item.append(3)
                    elif example_penalty["imprisonment"] > 3 * 12:
                         item.append(4)
                    elif example_penalty["imprisonment"] > 2 * 12:
                         item.append(5)
                    elif example_penalty["imprisonment"] > 1 * 12:
                         item.append(6)
                    elif example_penalty["imprisonment"] > 9:
                         item.append(7)
                    elif example_penalty["imprisonment"] > 6:
                         item.append(8)
                    elif example_penalty["imprisonment"] > 0:
                         item.append(9)
                    else:
                         item.append(10)
                    # 指控描述
                    list_str = json.dumps(item, ensure_ascii=False)
                    fw.write(list_str+"\n")
                    if count%5000==0:
                        print(f"已有{count}条数据被处理")
            fw.close()
        lang.update_label2index()
        # 将langs序列化
        f = open(os.path.join(data_base_path, folder, "lang.pkl"), "wb")
        pickle.dump(lang, f)
        f.close()

# 统计语料库
def getLang(folder):
    lang = Lang(folder)
    print("start statistic train data......")
    fr = open(os.path.join(data_base_path, folder, "train_d.json"), "r", encoding="utf-8")
    count = 0
    for line in fr:
        if line.strip() == "":
            continue
        count += 1
        item = json.loads(line)
        lang.addSentence(item[0])
        lang.addLabel(item[1])
        if count % 5000 == 0:
            print(f"已统计{count}条数据")
    lang.update_label2index()
    fr.close()
    # 序列化lang
    f = open("lang_data_train_preprocessed.pkl", "wb")
    pickle.dump(lang, f)
    f.close()
    print("train data statistic end.")

def make_accu2case_dataset(filename):
    accu2case = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item[1] not in accu2case:
                accu2case[item[1]] = [item[0]]
            else:
                accu2case[item[1]].append(item[0])
    return accu2case

def word2Index(file_path, lang, acc2id):
    # 数据集
    fi = open(file_path, "r", encoding="utf-8")
    fo = open(os.path.join(data_base_path, "data_train_forModel.txt"), "w", encoding="utf-8")
    count = 0
    for line in fi:
        count += 1
        # 文本数据
        item = json.loads(line)
        # 将文本妆化成索引
        item_num = []

        fact_num = []
        for fact in item[0:3]:
            fact_num.append([lang.word2index[word] for word in fact])
        item_num.append(fact_num[0])
        item_num.append(fact_num[1])
        item_num.append(fact_num[2])
        item_num.append(acc2id[item[3].strip()])
        item_num.append([lang.word2index[word] for word in item[4]])
        # 序列化并写入
        item_num_str = json.dumps(item_num, ensure_ascii=False)
        fo.write(item_num_str+"\n")
        if count%5000==0:
            print(f"已处理{count}条数据")
    fi.close()
    fo.close()

# 统计文本长度
def sample_length(path):
    max_length = 0
    max_length_sample = 0
    min_length = float("inf")
    min_length_sample = 0
    f = open(path, "r", encoding="utf-8")
    # all, 0-20, 20-50, 50-100, 100-200, 200-500, 500-1000, 1000-2000, 2000-5000, 5000-
    count = {"all":0, "0-20":0, "20-50":0,"50-100":0,"100-200":0,
             "200-500":0,"500-1000":0,"1000-2000":0,"2000-5000":0,
             "5000-":0}
    for line in f:
        count["all"] += 1
        sample = json.loads(line)
        length = len(sample[0])
        if length > max_length:
            max_length = length
            max_length_sample = count["all"]
        if length < min_length:
            min_length = length
            min_length_sample = count["all"]
        # 长度范围统计
        if length>=5000:
            count["5000-"] += 1
        else:
            if length>=200: # 200-5000
                if length>=1000: # 1000-5000
                    if length>=2000:
                        count["2000-5000"]+=1
                    else:
                        count["1000-2000"]+=1
                else: # 200-1000
                    if length>=500:
                        count["500-1000"]+=1
                    else:
                        count["200-500"]+=1
            else: # 0-200
                if length>=50: # 50-200
                    if length>=100:
                        count["100-200"]+=1
                    else:
                        count["50-100"]+=1
                else:
                    if length>=20:
                        count["20-50"]+=1
                    else:
                        count["0-20"]+=1
    f.close()
    return min_length, min_length_sample, max_length, max_length_sample, count

# 按照指控类别统计案件分布
def sample_categories_dis(file_path):
    f = open(file_path, "r", encoding="utf-8")
    acc_dict = {}
    for line in f:
        sample = json.loads(line)
        sample_acc = sample[3]
        if sample_acc not in acc_dict:
            acc_dict[sample_acc] = 1
        else:
            acc_dict[sample_acc] += 1
    f.close()
    return acc_dict

# load accusation classified
def load_classifiedAccus(filename):
    category2accu = {}
    accu2category = {}
    with open(filename, "r", encoding="utf-8") as f:
        count = 0
        for line in f:
            count+=1
            line = line.strip()
            item = line.split(" ")
            if item[0] not in category2accu:
                category2accu[item[0]] = item[1:]
            for accu in item[1:]:
                if accu not in accu2category:
                    accu2category[accu] = [item[0]]
                else:
                    accu2category[accu].append(item[0])
    return category2accu, accu2category

def val_test_datafilter(resourcefile, targetflie):
    # 根据训练数据过滤val和test数据集
    lang_f = open("lang_data_train_preprocessed.pkl", "rb")
    lang = pickle.load(lang_f)
    lang_f.close()
    fw = open(targetflie, "w", encoding="utf-8")
    print("start filter data......")
    with open(resourcefile, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            if len(example["meta"]["accusation"]) != 1:
                continue
            example_accu = example["meta"]["accusation"][0]
            if example_accu not in lang.label2index:
                continue
            else:
                # example_fact = example["fact"]
                # example_accu_idx = lang.label2index[example_accu]
                # example_str = json.dumps([example_fact, example_accu_idx],ensure_ascii=False)
                fw.write(line)
    fw.close()
    print("processing end ......")


if __name__=="__main__":
    # 过滤原始数据集
    # data_filter()
    #
    # langs = data_process()
    # print("end")
    accus_total = []
    f = open("../dataset/CAIL-SMALL/lang.pkl", "rb")
    lang_s = pickle.load(f)
    f.close()
    f = open("../dataset/CAIL-LARGE/lang.pkl", "rb")
    lang_l = pickle.load(f)
    f.close()
    accus_s = lang_s.index2accu
    accus_l = lang_l.index2accu
    accus_total.extend(accus_l)
    accus_total.extend(accus_s)
    accus_total = list(set(accus_total))
    with open("./accusation_classified_v2_2.txt", "r", encoding="utf-8") as f:
        accus = []
        for line in f:
            c = line.split(" ")
            accus.extend(c[1:])
        accus = list(set(accus))
    for i in accus_total:
        if i not in accus:
            print(i)
    # # 生成训练数据集
    # data_path = os.path.join(BATH_DATA_PATH, "data_train_filtered.json")
    # # acc_desc = commonUtils.get_acc_desc("accusation_description.json")
    # print("start processing data...")
    # getData(data_path, acc2desc=None)
    # print("data processing end.")
    #
    # 统计训练集语料库生成对象
    # lang_name = "2018_CAIL_SMALL_TRAIN"
    # getLang(lang_name)
    # f = open("lang_data_train_preprocessed.pkl", "rb")
    # lang = pickle.load(f)
    # print(lang.n_words)
    # print(lang.word2index['我'])

    # # # 将训练集中的文本转换成对应的索引
    # # print("start word to index")
    # id2acc, acc2id = getAccus(data_path)
    # f = open("lang_data_train_preprocessed.pkl", "rb")
    # lang = pickle.load(f)
    # f.close()
    # word2Index(os.path.join(BATH_DATA_PATH,"data_train_processed.txt"), lang, acc2id)
    # print("processing end")
    #
    # # 统计最长文本
    # print("start statistic length of sample......")
    # path = os.path.join(BATH_DATA_PATH, "data_train_processed.txt")
    # min_length,min_length_sample, max_length, max_length_sample, count = sample_length(path)
    # print(f"min_length: {min_length} at line {min_length_sample}")
    # print((f"max_length: {max_length} at line {max_length_sample}"))
    # print(count)
    #
    #
    # # 统计案件类别分布
    # file_path = os.path.join(BATH_DATA_PATH, "data_train_processed.txt")
    # sample_dis = sample_categories_dis(file_path)
    # f = open("sample_category_dis.pkl", "wb")
    # pickle.dump(sample_dis,f)
    # f.close()
    #
    # f = open("sample_category_dis.pkl", "rb")
    # sample_dis = pickle.load(f)
    # sample_dis = dict(sorted(sample_dis.items(), key=operator.itemgetter(1),reverse=True))
    # f.close()
    # print(sample_dis)
    #
    # top_10 = sum(list(sample_dis.values())[0:10])
    # bottom_10 = sum(list(sample_dis.values())[-10:])
    # print(f"top_10:{top_10/sum(sample_dis.values())}")
    # print(f"bottom_20:{bottom_10/sum(sample_dis.values())}")


    # # 获取指控字典
    # d1, d2 = getAccus(os.path.join(BATH_DATA_PATH,"data_train_filtered.json"))
    # print(len(d1))

    # category2accu, accu2category = load_classifiedAccus("accusation_classified_v2_1.txt")
    # print(len(category2accu))
    # print(len(accu2category))
    # print("end")


    # accu2case = make_accu2case_dataset("../dataset/CAIL-SMALL/data_train_processed.txt")
    # for _ in range(3):
    #     seq = commonUtils.pretrain_data_loader(accu2case, 8, 2)
    #     print(seq[0])
    #     print(seq[1])
    #     print("------------------xxxxx-------------------")
    # seq,label = commonUtils.pretrain_data_loader(accu2case=accu2case,
    #                                        batch_size=48,
    #                                        label2index=lang.label2index,
    #                                        positive_size=2,
    #                                        sim_accu_num=4,
    #                                        category2accu=category2accu,
    #                                        accu2category=accu2category)
    #
    # print(seq)
    # resourcefile = "../dataset/CAIL-SMALL/data_valid_filtered.json"
    # targetfile = "../dataset/CAIL-SMALL/data_valid_processed.txt"
    # # val_test_datafilter(resourcefile, targetfile)
    # getData(resourcefile, targetfile)











