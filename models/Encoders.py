import torch
import torch.nn as nn
from transformers import BertModel

class GRUEnc(nn.Module):
    def __init__(self):
        super(GRUEnc, self).__init__()




class BERTEnc(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(BERTEnc, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-chinese",
                                              num_labels=label_size,
                                              output_attentions=False,
                                              output_hidden_states=True)

        self.contraLinear = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.BatchNorm1d(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

        self.chargePreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, label_size)
        )



    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        hidden_state = torch.mean(outputs.last_hidden_state, dim=1)
        # [batch_size, hidden_size] -> [batch_size, hidden_size]
        contra_hiddens = self.contraLinear(hidden_state)
        # [batch_size, hidden_size] -> [batch_size, label_size]
        classify_preds = self.chargePreds(contra_hiddens)
        # [batch_size, hidden_size]、[batch_size, label_size]
        return contra_hiddens, classify_preds



class ChargeAwareAttention(nn.Module):

    def __init__(self, hidden_size):
        super(ChargeAwareAttention, self).__init__()
        # 对contras_vecs 做线性变换
        self.contras_vecs_trans = nn.Sequential(
            nn.Linear(hidden_size, 2*hidden_size),
            nn.BatchNorm1d(2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size)
        )





    def forward(self, encs, contra_vecs):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param contra_vecs: vec train by supervised contrastive learning --> [batch_size, hidden_size]
        :return:art_vecs for law article preds --> [batch_size, hidden_size]
        """
        # [batch_size, hidden_size] --> [batch_size, hidden_size]
        contra_vecs = self.contras_vecs_trans(contra_vecs)
        # [batch_size, seq_length, hidden_size]*[batch_size, hidden_size, 1] -->

