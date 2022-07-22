import torch
import pickle
import json
import configparser
import torch.nn as nn
import torch.optim as optim
from models.models import GRULJP
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from utils.commonUtils import pretrain_data_loader, train_distloss_fun, penalty_constrain, Lang

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("load config file")
section = "gru-train"
config = configparser.ConfigParser()
config.read('../config.cfg')
EPOCH = int(config.get(section, "EPOCH"))
BATCH_SIZE = int(config.get(section, "BATCH_SIZE"))
HIDDEN_SIZE = int(config.get(section, "HIDDEN_SIZE"))
POSITIVE_SIZE = int(config.get(section, "POSITIVE_SIZE"))
SIM_ACCU_NUM = int(config.get(section, "SIM_ACCU_NUM"))
PENALTY_LABEL_SIZE = int(config.get(section,"PENALTY_LABEL_SIZE"))
LR = float(config.get(section, "LR"))
STEP = int(config.get(section, "STEP"))
CHARGE_RADIUS = int(config.get(section, "CHARGE_RADIUS"))
PENALTY_RADIUS = int(config.get(section, "PENALTY_RADIUS"))

corpus_info_path = "../dataprepare/lang-CAIL-SMALL.pkl"
data_path = "../dataset/CAIL-SMALL/train_processed.txt"
accu_similarity = "../dataprepare/accusation_classified_v2_2.txt"

print("load corpus info")
f = open(corpus_info_path, "rb")
lang = pickle.load(f)
f.close()

print("load dataset classified by accusation")
accu2case = make_accu2case_dataset(data_path, lang=lang, input_idx=1, accu_idx=2)

print("load accusation similarity sheet")
category2accu, accu2category = load_classifiedAccus(accu_similarity)

model = GRULJP(charge_label_size=len(lang.index2accu),
               article_label_size=len(lang.index2art),
               penalty_label_size=PENALTY_LABEL_SIZE,
               voc_size=lang.n_words,
               hidden_size=HIDDEN_SIZE)
model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器 AdamW由Transfomer提供,目前看来表现很好
# optimizer = AdamW(model.parameters(),
#                   lr=LR,
#                   eps=1e-8)
optimizer = optim.Adam(model.parameters(), lr=LR)

print("gru based model train start......\n")

train_loss = 0
train_loss_records = []
valid_loss_records = []
valid_acc_records = []
valid_mp_records = []
valid_f1_records = []
valid_mr_records = []
for step in range(STEP):
    # 随机生成一个batch
    if step%EPOCH == 0:
        start = timer()
    start = timer()
    seqs, accu_labels, article_labels, penalty_labels = pretrain_data_loader(accu2case=accu2case,
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
    accu_labels = torch.tensor(accu_labels)
    accu_labels = torch.cat(accu_labels, dim=0)
    charge_preds_loss = criterion(charge_preds_outputs, accu_labels)

    # 法律条款预测误差
    article_preds_outputs = torch.cat(article_preds_outputs, dim=0)
    article_labels = torch.tensor(article_labels)
    article_labels = torch.cat(article_labels, dim=0)
    article_preds_loss = criterion(article_preds_outputs, article_labels)

    # 刑期预测结果约束（相似案件的刑期应该相近）
    penalty_contrains = torch.torch.stack(penalty_preds_outputs, dim=0)
    penalty_contrains_loss = penalty_constrain(penalty_contrains, PENALTY_RADIUS)


    # 刑期预测误差
    penalty_preds_outputs = torch.cat(penalty_preds_outputs, dim=0)
    penalty_labels = torch.tensor(penalty_labels)
    penalty_labels = torch.cat(penalty_labels, dim=0)
    penalty_preds_loss = criterion(penalty_preds_outputs, penalty_labels)

    loss = posi_pairs_dist+neg_pairs_dist+charge_preds_loss+article_preds_loss+penalty_preds_loss
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

        train_loss_records.append(train_loss / EPOCH)

        end = timer()
        print(f"Epoch: {int((step + 1)/EPOCH)}  Train_loss: {round(train_loss/EPOCH, 6)}  Valid_loss: {round(valid_loss,6)}   Accuracy: {round(accuracy, 6)}  \n"
              f"F1: {round(f1, 6)}  MR: {round(mr, 6)}  MP: {round(mp, 6)}  Time: {round((end-start)/60, 2)}min \n")

        # 保存模型
        save_path = f"../dataset/model_checkpoints/round-1-1/model_at_epoch-{int((step + 1)/EPOCH)}_.pt"
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