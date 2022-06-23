from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup, AdamW
from utils.commonUtils import pretrain_data_loader, train_distloss_fun, Lang
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from models.bert_base import ContrasBert
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

# seed_val = 42
#
# random.seed(seed_val)
# np.random.seed(seed_val)
# torch.manual_seed(seed_val)
# torch.cuda.manual_seed_all(seed_val)

bert_hidden_size = 768
LABEL_SIZE = 112
EPOCH = 2000*10
BATCH_SIZE = 12
POSI_SIZE = 2
SIM_ACCU_NUM = 3
LR = 0.0001
M = 10

# model = BertModel.from_pretrained("bert-base-chinese")
# model = BertForSequenceClassification.from_pretrained("bert-base-chinese",
#                                                       num_labels = 112,
#                                                       output_attentions = False,
#                                                       output_hidden_states = True)
model = ContrasBert(hidden_size=bert_hidden_size, label_size=LABEL_SIZE)
model.to(device)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器
# optimizer = optim.SGD(model.parameters(), lr=LR)
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)

# 学习率优化策略
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = EPOCH)
print("fine-tune start......")
for epoch in range(EPOCH):

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
    classfy_loss = criterion(classify_outputs, label_ids)

    loss = posi_pairs_dist+neg_pairs_dist+classfy_loss

    # 反向传播计算梯度
    loss.backward()

    # 梯度裁剪防止梯度爆炸
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # 更新梯度
    optimizer.step()

    # 更新学习率
    scheduler.step()

    if (epoch+1)%100 == 0:
        print(f"step: {epoch+1}    loss: {loss} \n")




