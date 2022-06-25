from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AdamW
from utils.commonUtils import pretrain_data_loader, train_distloss_fun, Lang, \
    data_loader, accumulated_accuracy, prepare_valid_data, ConfusionMatrix
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from models.bert_base import ContrasBert
from timeit import default_timer as timer
import torch
import random
import pickle
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accu2case = make_accu2case_dataset("../dataset/CAIL-SMALL/data_train_processed.txt")
category2accu, accu2category = load_classifiedAccus("../dataprepare/accusation_classified_v2_1.txt")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
f = open("../dataprepare/lang_data_train_preprocessed.pkl", "rb")
lang = pickle.load(f)
f.close()

bert_hidden_size = 768
EPOCH = 2000
LABEL_SIZE = 112
STEP = EPOCH*200
BATCH_SIZE = 12
POSI_SIZE = 2
SIM_ACCU_NUM = 3
LR = 0.0001
M = 10

# model = ContrasBert(hidden_size=bert_hidden_size, label_size=LABEL_SIZE)
# model = model.to(device)
model = torch.load("../dataset/model_checkpoints/round-1-1/model_at_epoch-50_.pkl", map_location=device)

# 设置数据并行
# model = nn.DataParallel(model)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器 AdamW由Transfomer提供,目前看来表现很好
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)

# 学习率优化策略
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = STEP)

train_loss_records = []
valid_loss_records = []
valid_acc_records = []
valid_mp_records = []
valid_f1_records = []
valid_mr_records = []

# 初始化混淆矩阵
confusMat = ConfusionMatrix(len(lang.index2label))
# 验证模型在验证集上的表现
model.eval()
valid_loss = 0
val_step = 0
valid_seq, valid_label = prepare_valid_data("../dataset/CAIL-SMALL/data_valid_processed.txt")
for val_seq, val_label in data_loader(valid_seq, valid_label, batch_size=BATCH_SIZE):
    val_label = torch.tensor(val_label, dtype=torch.long).to(device)
    val_seq_enc = tokenizer.batch_encode_plus(val_seq,
                                            add_special_tokens=False,
                                            max_length=512,
                                            truncation=True,
                                            padding=True,
                                            return_attention_mask=True,
                                            return_tensors='pt')
    with torch.no_grad():
        val_contra_hidden, val_classify_preds = model(input_ids=val_seq_enc["input_ids"].to(device),
                                              attention_mask=val_seq_enc["attention_mask"].to(device))
        val_classify_loss = criterion(val_classify_preds, val_label)
        valid_loss += val_classify_loss.item()
        right_preds, batch_len = accumulated_accuracy(val_classify_preds.cpu().numpy(), val_label.cpu().numpy())
        confusMat.updateMat(val_classify_preds.cpu().numpy(), val_label.cpu().numpy())
    val_step += 1

valid_loss = valid_loss/val_step
valid_loss_records.append(valid_loss)

accuracy = confusMat.get_acc() # 根据混淆矩阵求解acc
valid_acc_records.append(accuracy)

f1 = confusMat.getMaF()
valid_f1_records.append(f1)

mr = confusMat.getMaR()
valid_mr_records.append(mr)

mp = confusMat.getMaP()
valid_mp_records.append(mp)

print(f"Valid_loss: {round(valid_loss,6)}   Accuracy: {round(accuracy, 6)}  \n"
      f"F1: {round(f1, 6)}  MR: {round(mr, 6)}  MP: {round(mp, 6)} \n")

