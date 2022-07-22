import torch
from models import GRULJP

gru = GRULJP(voc_size=10, hidden_size=8, num_layers=2, dropout=0.5, charge_label_size=3, article_label_size=3, penalty_label_size=3)

input_ids = [torch.tensor([1,2,4]).long(),torch.tensor([3,2]).long()]

encs = gru(input_ids)
print(encs)

