import torch
import pickle
import configparser
import torch.nn as nn
import torch.optim as optim
from models.models import GRULJP
from timeit import default_timer as timer
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from utils.commonUtils import pretrain_data_loader, train_distloss_fun, Lang

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
PENALTY_LABEL_SIZE = int(config.get(section,"penalty_label_size"))
LR = float(config.get(section, "LR"))
STEP = int(config.get(section, "STEP"))


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
    seq, label_ids = pretrain_data_loader(accu2case=accu2case,
                                          batch_size=BATCH_SIZE,
                                          lang=lang,
                                          positive_size=POSITIVE_SIZE,
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