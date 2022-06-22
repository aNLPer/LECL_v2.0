from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from utils.commonUtils import pretrain_data_loader, train_distloss_fun
from dataprepare.dataprepare import make_accu2case_dataset, load_classifiedAccus
import torch
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
accu2case = make_accu2case_dataset("../dataset/CAIL-SMALL/data_train_processed.txt")
category2accu, accu2category = load_classifiedAccus("../dataprepare/accusation_classified_v2_1.txt")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

EPOCH = 1000
BATCH_SIZE = 72
POSI_SIZE = 2
SIM_ACCU_NUM = 4
LR = 0.01

# model = BertModel.from_pretrained("bert-base-chinese")
model = BertForSequenceClassification.from_pretrained("Bert-base-chinese",
                                                      num_labels = 112,
                                                      output_attentions = False,
                                                      output_hidden_states = True)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    print("fine-tune start......")
    seq = pretrain_data_loader(accu2case=accu2case,
                                    batch_size=BATCH_SIZE,
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

    batch_enc_ids = torch.stack(batch_enc_ids, dim=0)
    batch_enc_ids = batch_enc_ids.to(device)
    batch_enc_atten_mask = torch.stack(batch_enc_atten_mask,dim=0)
    batch_enc_atten_mask = batch_enc_atten_mask.to(device)

    with torch.no_grad():
        outputs = []
        for i in range(POSI_SIZE):
            output = model(input_ids=batch_enc_ids[i], attention_mask=batch_enc_atten_mask[i])
            outputs.append(torch.mean(output.last_hidden_state, dim=1))
        outputs = torch.stack(outputs, dim=0)

        loss_dist = train_distloss_fun(outputs)

        loss_dist.backward()


