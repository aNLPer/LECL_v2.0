from transformers import BertModel, BertTokenizer
from utils.commonUtils import pretrain_data_loader
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

model = BertModel.from_pretrained("bert-base-chinese")
model.to(device)

# optimizer = optim.SGD(model.parameters(), lr=LR)

for epoch in range(EPOCH):
    print("fine-tune start......")
    seq = pretrain_data_loader(accu2case=accu2case,
                                    batch_size=BATCH_SIZE,
                                    positive_size=POSI_SIZE,
                                    sim_accu_num=SIM_ACCU_NUM,
                                    category2accu=category2accu,
                                    accu2category=accu2category)
    batch_enc_ids = []
    batch_enc_attn_mask = []
    print("max_length:",max([len(s) for s in seq[0]]))
    for i in range(POSI_SIZE):
        batch_enc = tokenizer.batch_encode_plus(seq[i],
                                    add_special_tokens=False,
                                    max_length=512,
                                    truncation=True,
                                    padding=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')
        batch_enc_ids.append(batch_enc["input_ids"])
        batch_enc_attn_mask.append(batch_enc["attention_mask"])

    output = model(input_ids=batch_enc_ids[0], attention_mask=batch_enc_attn_mask[0])
    print(output.shape)