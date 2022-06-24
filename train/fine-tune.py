from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AdamW
from utils.commonUtils import pretrain_data_loader, train_distloss_fun, Lang, data_loader
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
STEP = EPOCH*30
BATCH_SIZE = 12
POSI_SIZE = 2
SIM_ACCU_NUM = 3
LR = 0.0001
M = 10

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

def accumulated_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat), len(labels_flat)


model = ContrasBert(hidden_size=bert_hidden_size, label_size=LABEL_SIZE)
model = model.to(device)
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
print("fine-tune start......")
train_loss = 0
train_loss_records = []
valid_loss_records = []
valid_acc_records = []
for step in range(STEP):
    # 随机生成一个batch
    if step%EPOCH == 0:
        start = timer()
    start = timer()
    seq, label_ids = pretrain_data_loader(accu2case=accu2case,
                                          batch_size=BATCH_SIZE,
                                          label2index=lang.label2index,
                                          positive_size=POSI_SIZE,
                                          sim_accu_num=SIM_ACCU_NUM,
                                          category2accu=category2accu,
                                          accu2category=accu2category)
    batch_enc_ids = []
    batch_enc_atten_mask = []
    for i in range(POSI_SIZE):
        batch_enc = tokenizer.batch_encode_plus(seq[i],
                                    add_special_tokens=False,
                                    max_length=512,
                                    truncation=True,
                                    padding=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
        batch_enc_ids.append(batch_enc["input_ids"])
        batch_enc_atten_mask.append(batch_enc["attention_mask"])

    # 设置模型状态
    model.train()

    # 优化参数的梯度置0
    optimizer.zero_grad()

    # 计算模型的输出
    contra_outputs = []
    classify_outputs = []
    for i in range(POSI_SIZE):
        # [batch_size/2, hidden_size]、[batch_size/2, label_size]
        contra_hidden, classify_preds = model(input_ids=batch_enc_ids[i].to(device),
                                              attention_mask=batch_enc_atten_mask[i].to(device))
        contra_outputs.append(contra_hidden)
        classify_outputs.append(classify_preds)

    # 计算误差
    contra_outputs = torch.stack(contra_outputs, dim=0)  # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
    classify_outputs = torch.cat(classify_outputs, dim=0)  # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
    label_ids = torch.tensor(label_ids * 2).to(device) # [batch_size,]
    posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=M)
    classify_loss = criterion(classify_outputs, label_ids)

    loss = posi_pairs_dist+neg_pairs_dist+classify_loss
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

        # 验证模型在验证集上的表现
        model.eval()
        valid_loss = 0
        total_eval_accuracy = 0
        length_val_data = 0
        valid_seq, valid_label = prepare_valid_data("../dataset/CAIL-SMALL/data_valid_processed.txt")
        for val_seq, val_label in data_loader(valid_seq, valid_label, batch_size=BATCH_SIZE):
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
            right_preds, batch_len = accumulated_accuracy(classify_preds.numpy(), np.array(val_label))
            total_eval_accuracy += right_preds
            length_val_data += batch_len

        valid_loss = valid_loss/length_val_data
        valid_loss_records.append(valid_loss)

        accuracy = total_eval_accuracy/length_val_data
        valid_acc_records.append(accuracy)

        train_loss_records.append(train_loss / EPOCH)
        train_loss = 0
        end = timer()
        print(f"epoch: {(step + 1)/EPOCH}  train_loss: {train_loss / EPOCH}  valid_loss: {valid_loss}  accuracy: {accuracy}  time: {(end - start) / 60}min")

train_loss_records = json.dumps(train_loss_records, ensure_ascii=False)
valid_loss_records = json.dumps(valid_loss_records, ensure_ascii=False)
valid_acc_records = json.dumps(valid_acc_records,ensure_ascii=False)
with open("./training_records.txt", "w", encoding="utf-8") as f:
    f.write('train_loss_records\t' + train_loss_records + "\n")
    f.write('valid_loss_records\t' + valid_loss_records + "\n")
    f.write('valid_acc_records\t' + valid_acc_records + "\n")