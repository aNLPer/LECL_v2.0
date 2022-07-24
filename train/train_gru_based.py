import torch
import pickle
import json
import gensim
import configparser
import torch.nn as nn
import numpy as np
import torch.optim as optim
from models.models import GRULJP
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, AdamW
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from utils.commonUtils import contras_data_loader, train_distloss_fun, penalty_constrain, ConfusionMatrix, prepare_valid_data, data_loader, check_data, Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("load config file")
section = "gru-train"
config = configparser.ConfigParser()
config.read('../config.cfg')

EPOCH = int(config.get(section, "EPOCH"))
BATCH_SIZE = int(config.get(section, "BATCH_SIZE"))
HIDDEN_SIZE = int(config.get(section, "HIDDEN_SIZE"))
POSITIVE_SIZE = int(config.get(section, "POSITIVE_SIZE"))
MAX_LENGTH = int(config.get(section, "MAX_LENGTH"))
SIM_ACCU_NUM = int(config.get(section, "SIM_ACCU_NUM"))
PENALTY_LABEL_SIZE = int(config.get(section,"PENALTY_LABEL_SIZE"))
LR = float(config.get(section, "LR"))
STEP = int(config.get(section, "STEP"))
CHARGE_RADIUS = int(config.get(section, "CHARGE_RADIUS"))
PENALTY_RADIUS = int(config.get(section, "PENALTY_RADIUS"))
LAMDA = float(config.get(section, "LAMDA")) # 刑期约束系数
ALPHA = float(config.get(section, "ALPHA")) #
GRU_LAYERS = int(config.get(section, 'GRU_LAYERS'))
DROPOUT_RATE = float(config.get(section, "DROPOUT_RATE"))
L2 = float(config.get(section, "L2"))


corpus_info_path = "../dataprepare/lang-CAIL-SMALL-word-level.pkl"
data_path = "../dataset/CAIL-SMALL/train_processed.txt"
accu_similarity = "../dataprepare/accusation_classified_v2_2.txt"

print("load corpus info")
f = open(corpus_info_path, "rb")
lang = pickle.load(f)
f.close()

print("load pretrained word2vec")
pretrained_model = gensim.models.KeyedVectors.load_word2vec_format('../dataset/token_vec_300.bin', binary=False)

print("load dataset classified by accusation")
accu2case = make_accu2case_dataset(data_path, lang=lang, input_idx=0, accu_idx=2, max_length=MAX_LENGTH, pretrained_vec=pretrained_model)

print("load accusation similarity sheet")
category2accu, accu2category = load_classifiedAccus(accu_similarity)




model = GRULJP(charge_label_size=len(lang.index2accu),
               article_label_size=len(lang.index2art),
               penalty_label_size=PENALTY_LABEL_SIZE,
               voc_size=lang.n_words,
               dropout=DROPOUT_RATE,
               num_layers=GRU_LAYERS,
               hidden_size=HIDDEN_SIZE,
               pretrained_model = pretrained_model,
               mode="concat")

# model =
model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器 AdamW由Transfomer提供,目前看来表现很好
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=L2)
optimizer = optim.AdamW([{"params":model.em.parameters(),'lr':1e-3}], lr=LR)
# optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
# optimizer = optim.SGD(model.parameters(), lr=LR)

# 学习率优化策略
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 500, # Default value in run_glue.py
                                            num_training_steps = STEP)


print("gru based model train start......\n")

train_loss = 0
train_loss_records = []
valid_loss_records = []
valid_acc_records = {"charge":[], "article":[], "penalty":[]}
valid_mp_records = {"charge":[], "article":[], "penalty":[]}
valid_f1_records = {"charge":[], "article":[], "penalty":[]}
valid_mr_records = {"charge":[], "article":[], "penalty":[]}
for step in range(STEP):
    # 随机生成一个batch
    if step%EPOCH == 0:
        start = timer()
    start = timer()
    seqs, accu_labels, article_labels, penalty_labels = contras_data_loader(accu2case=accu2case,
                                          batch_size=BATCH_SIZE,
                                          lang=lang,
                                          positive_size=POSITIVE_SIZE,
                                          sim_accu_num=SIM_ACCU_NUM,
                                          category2accu=category2accu,
                                          accu2category=accu2category)
    # 设置模型状态
    model.train()

    # 优化参数的梯度置0
    optimizer.zero_grad()

    # 计算模型的输出
    charge_vecs_outputs = []
    charge_preds_outputs = []
    article_preds_outputs = []
    penalty_preds_outputs = []
    for i in range(POSITIVE_SIZE):
        # [batch_size/2, hidden_size]、[batch_size/2, label_size]
        seq_lens = []
        for tensor in seqs[i]:
            seq_lens.append(tensor.shape[0])
        padded_input_ids = pad_sequence(seqs[i], batch_first=True).to(device)

        charge_vecs, charge_preds, article_preds, penalty_preds = model(padded_input_ids, seq_lens)
        charge_vecs_outputs.append(charge_vecs)
        charge_preds_outputs.append(charge_preds)
        article_preds_outputs.append(article_preds)
        penalty_preds_outputs.append(penalty_preds)

    # charge_vecs的对比误差
    contra_outputs = torch.stack(charge_vecs_outputs, dim=0)  # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
    posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=CHARGE_RADIUS)

    # 指控分类误差
    charge_preds_outputs = torch.cat(charge_preds_outputs, dim=0)  # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
    accu_labels = [torch.tensor(l) for l in accu_labels]
    accu_labels = torch.cat(accu_labels, dim=0).to(device)
    charge_preds_loss = criterion(charge_preds_outputs, accu_labels)

    # 法律条款预测误差
    article_preds_outputs = torch.cat(article_preds_outputs, dim=0)
    article_labels = [torch.tensor(l) for l in article_labels]
    article_labels = torch.cat(article_labels, dim=0).to(device)
    article_preds_loss = criterion(article_preds_outputs, article_labels)

    # 刑期预测结果约束（相似案件的刑期应该相近）
    penalty_contrains = torch.stack(penalty_preds_outputs, dim=0).to(device)
    penalty_contrains_loss = penalty_constrain(penalty_contrains, PENALTY_RADIUS)

    # 刑期预测误差
    penalty_preds_outputs = torch.cat(penalty_preds_outputs, dim=0)
    penalty_labels = [torch.tensor(l) for l in penalty_labels]
    penalty_labels = torch.cat(penalty_labels, dim=0).to(device)
    penalty_preds_loss = criterion(penalty_preds_outputs, penalty_labels)

    loss = ALPHA * (posi_pairs_dist - neg_pairs_dist) + \
           charge_preds_loss + article_preds_loss+ penalty_preds_loss + \
           LAMDA * penalty_contrains_loss
    train_loss += loss.item()

    # 反向传播计算梯度
    loss.backward()

    # 梯度裁剪防止梯度爆炸
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 更新梯度
    optimizer.step()

    # 更新学习率
    scheduler.step()

    # 训练完一个EPOCH后评价模型
    if (step+1)%EPOCH == 0:
        # 初始化混淆矩阵
        charge_confusMat = ConfusionMatrix(len(lang.index2accu))
        article_confusMat = ConfusionMatrix(len(lang.index2art))
        penalty_confusMat = ConfusionMatrix(PENALTY_LABEL_SIZE)
        # 验证模型在验证集上的表现
        model.eval()
        valid_loss = 0
        val_step = 0
        valid_seq, valid_charge_labels, valid_article_labels, valid_penalty_labels = \
            prepare_valid_data("../dataset/CAIL-SMALL/test_processed.txt", lang,input_idx=0, max_length=MAX_LENGTH, pretrained_model=pretrained_model)

        for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(valid_seq, valid_charge_labels, valid_article_labels, valid_penalty_labels, batch_size=BATCH_SIZE):
            val_seq_lens = [len(s) for s in val_seq]
            val_input_ids = [torch.tensor(s) for s in val_seq]
            val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(device)
            with torch.no_grad():
                val_charge_vecs, val_charge_preds, val_article_preds, val_penalty_preds = model(val_input_ids, val_seq_lens)
                val_charge_preds_loss = criterion(val_charge_preds, torch.tensor(val_charge_label).to(device))
                val_article_preds_loss = criterion(val_article_preds, torch.tensor(val_article_label).to(device))
                val_penalty_preds_loss = criterion(val_penalty_preds, torch.tensor(val_penalty_label).to(device))
                valid_loss += val_charge_preds_loss.item()
                valid_loss += val_article_preds_loss.item()
                valid_loss += val_penalty_preds_loss.item()
                charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))
            val_step += 1

        train_loss_records.append(train_loss / EPOCH)

        valid_loss = valid_loss/val_step*BATCH_SIZE
        valid_loss_records.append(valid_loss)

        # acc
        valid_acc_records['charge'].append(charge_confusMat.get_acc())
        valid_acc_records['article'].append(article_confusMat.get_acc())
        valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

        # F1
        valid_f1_records['charge'].append(charge_confusMat.getMaF())
        valid_f1_records['article'].append(article_confusMat.getMaF())
        valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

        # MR
        valid_mr_records['charge'].append(charge_confusMat.getMaR())
        valid_mr_records['article'].append(article_confusMat.getMaR())
        valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

        # MP
        valid_mp_records['charge'].append(charge_confusMat.getMaP())
        valid_mp_records['article'].append(article_confusMat.getMaP())
        valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

        end = timer()
        print(f"Epoch: {int((step + 1)/EPOCH)}  Train_loss: {round(train_loss/EPOCH, 6)}  Valid_loss: {round(valid_loss,6)} \n"
              f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}\n"
              f"Article_Acc: {round(article_confusMat.get_acc(), 6)}  Article_F1: {round(article_confusMat.getMaF(), 6)}  Article_MR: {round(article_confusMat.getMaR(), 6)}  Article_MP: {round(article_confusMat.getMaP(), 6)}\n"
              f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 6)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 6)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 6)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 6)}\n"
              f"Time: {round((end-start)/60, 2)}min ")

        # 保存模型
        save_path = f"../dataset/checkpoints/model-at-epoch-{int((step + 1)/EPOCH)}_.pt"
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler': scheduler.state_dict()
        }, save_path)

        train_loss = 0

train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
valid_acc_records = json.dumps(valid_acc_records,ensure_ascii=False)
valid_mp_records = json.dumps(valid_mp_records,ensure_ascii=False)
valid_f1_records = json.dumps(valid_f1_records,ensure_ascii=False)
valid_mr_records = json.dumps(valid_mr_records,ensure_ascii=False)
with open(f"./training_records.txt", "w", encoding="utf-8") as f:
    f.write('train_loss_records\t' + train_loss_records + "\n")
    f.write('valid_loss_records\t' + valid_loss_records + "\n")
    f.write('valid_acc_records\t' + valid_acc_records + "\n")
    f.write('valid_mp_records\t' + valid_mp_records + "\n")
    f.write('valid_f1_records\t' + valid_f1_records + "\n")
    f.write('valid_mr_records\t' + valid_mr_records + "\n")