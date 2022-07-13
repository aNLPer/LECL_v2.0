import torch
import torch.nn as nn
from transformers import BertModel

class GRULJPred(nn.Module):
    def __init__(self):
        super(GRULJPred, self).__init__()




class BERTLJPred(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(BERTLJPred, self).__init__()
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



class ChargeAwareAtten(nn.Module):
    def __init__(self, hidden_size):
        """
        :param hidden_size: 隐变量的维度
        """
        super(ChargeAwareAtten, self).__init__()
        # encoder 输出的转换矩阵
        self.transM = torch.normal(0, 1, (hidden_size, hidden_size),
                                   dtype=torch.float32,
                                   requires_grad=True)

    def forward(self, encs, contra_vecs, mode="concat"):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param contra_vecs: vec train by supervised contrastive learning --> [batch_size, hidden_size]
        :return:art_vecs for law article preds --> [batch_size, hidden_size]
        """
        # [batch_size, seq_length, hidden_size]
        encs_transed = torch.matmul(encs, self.transM)
        # [batch_size, hidden_size, 1]
        contra_vecs = torch.unsqueeze(contra_vecs, dim=2)

        # [batch_size, seq_length, 1]
        atten_score = torch.matmul(encs_transed, contra_vecs)

        # [batch_size, seq_length, 1]
        atten_score = atten_score.softmax(dim=1)

        # [batch_size, hidden_size, seq_length]
        encs_transed = encs_transed.transpose(dim0=1, dim1=2)

        # [batch_size, hidden_size, 1]
        atten_sum = torch.matmul(encs_transed, atten_score)

        # [batch_size, hidden_size]
        atten_sum = atten_sum.squeeze()

        # [batch_size, 2*hidden_size]
        if mode == "concat":
            return torch.concat((atten_sum, contra_vecs), dim=1)
        if mode == "sum":
            return atten_sum + contra_vecs

class ArticleAwareAtten(nn.Module):
    def __init__(self, hidden_size):
        """
        :param hidden_size: 隐变量的维度
        """
        super(ArticleAwareAtten, self).__init__()
        # encoder 输出的转换矩阵
        self.hidden_size = hidden_size
        self.transM = torch.normal(0, 1, (self.hidden_size, self.hidden_size),
                                   dtype=torch.float32,
                                   requires_grad=True)

        self.transArt = None

    def forward(self, encs, art_vecs, mode="concat"):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param contra_vecs: vec train by supervised contrastive learning --> [batch_size, hidden_size]
        :return:art_vecs for law article preds --> [batch_size, hidden_size]
        """

        art_hidden_size = art_vecs.size()[1]
        self.transArt = torch.normal(0,1, (art_hidden_size, self.hidden_size),
                                     dtype=torch.float32,
                                     requires_grad=True)

        # [batch_size, seq_length, hidden_size]
        encs_transed = torch.matmul(encs, self.transM)
        # [batch_size, hidden_size]
        art_vecs_transed = torch.matmul(art_vecs, self.transArt)

        # [batch_size, hidden_size, 1]
        art_vecs_transed = torch.unsqueeze(art_vecs_transed, dim=2)

        # [batch_size, seq_length, 1]
        atten_score = torch.matmul(encs_transed, art_vecs_transed)

        # [batch_size, seq_length, 1]
        atten_score = atten_score.softmax(dim=1)

        # [batch_size, hidden_size, seq_length]
        encs_transed = encs_transed.transpose(dim0=1, dim1=2)

        # [batch_size, hidden_size, 1]
        atten_sum = torch.matmul(encs_transed, atten_score)

        # [batch_size, hidden_size]
        atten_sum = atten_sum.squeeze()
        if mode == "concat":
            return torch.concat((atten_sum, art_vecs_transed), dim=1)
        if mode == "sum":
            return atten_sum + art_vecs_transed