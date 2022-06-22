from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from utils.commonUtils import pretrain_data_loader, train_distloss_fun
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
from models.bert_base import ContrasBert
import torch
import pickle
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
LABEL_SIZE = 112
EPOCH = 1000
BATCH_SIZE = 72
POSI_SIZE = 2
SIM_ACCU_NUM = 4
LR = 0.01
M = 10

# model = BertModel.from_pretrained("bert-base-chinese")
# model = BertForSequenceClassification.from_pretrained("bert-base-chinese",
#                                                       num_labels = 112,
#                                                       output_attentions = False,
#                                                       output_hidden_states = True)
model = ContrasBert(hidden_size=bert_hidden_size, label_size=LABEL_SIZE)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    print("fine-tune start......")

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

    # batch_enc_ids = torch.stack(batch_enc_ids, dim=0)
    # batch_enc_ids = batch_enc_ids.to(device)
    # batch_enc_atten_mask = torch.stack(batch_enc_atten_mask,dim=0)
    # batch_enc_atten_mask = batch_enc_atten_mask.to(device)

    with torch.no_grad():
        contra_outputs = []
        classify_outputs = []
        for i in range(POSI_SIZE):
            # [batch_size/2, hidden_size]、[batch_size/2, label_size]
            contra_hidden, classify_preds = model(input_ids=batch_enc_ids[i].to(device), attention_mask=batch_enc_atten_mask[i].to(device))
            contra_outputs.append(contra_hidden)
            classify_outputs.append(classify_preds)
        # 2 * [batch_size/posi_size, hidden_size] -> [posi_size, batch_size/posi_size, hidden_size]
        contra_outputs = torch.stack(contra_outputs, dim=0)
        # [posi_size, batch_size/posi_size, label_size] -> [batch_size, label_size]
        classify_outputs = torch.cat(classify_outputs, dim=0)
        # [batch_size,]
        label_ids = torch.tensor(label_ids * 2).to(device)

        posi_pairs_dist, neg_pairs_dist = train_distloss_fun(contra_outputs, radius=M)
        classfy_loss = criterion(classify_outputs, label_ids)

        loss = posi_pairs_dist+neg_pairs_dist+classfy_loss

        loss.backward()

        print(posi_pairs_dist, neg_pairs_dist)


