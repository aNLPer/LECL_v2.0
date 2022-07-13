import torch
import torch.nn as nn
from transformers import BertModel
# distinguish confusing charge for legal judgement prediction


class GRULJPred(nn.Module):
    def __init__(self):
        super(GRULJPred, self).__init__()


class BERTLJPred(nn.Module):
    def __init__(self, hidden_size,
                 charge_label_size,
                 article_label_size,
                 penalty_label_size,
                 enc_type="bert",
                 mode="concat"):
        super(BERTLJPred, self).__init__()
        self.enc_mode = enc_type
        if self.enc_mode == "bert":
            self.enc = BertModel.from_pretrained("bert-base-chinese",
                                                  num_labels=charge_label_size,
                                                  output_attentions=False,
                                                  output_hidden_states=True)
        if self.enc_mode == "gru":
            pass

        self.chargeAwareAtten = CharacAwareAtten(hidden_size, mode)

        self.articleAwareAtten = CharacAwareAtten(hidden_size, mode)

        self.chargeLinear = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.BatchNorm1d(4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )

        self.chargePreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, charge_label_size)
        )

        self.articlePreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, article_label_size)
        )

        self.penaltyPreds = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, penalty_label_size)
        )


    def forward(self, input_ids, attention_mask):
        # [batch_size, seq_length, hidden_size]
        if self.enc_mode == "bert":
            encs = self.enc(input_ids=input_ids, attention_mask=attention_mask)  # encoder 输出
        if self.enc_mode == "gru":
            encs = None
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        hidden_state = torch.mean(encs.last_hidden_state, dim=1)
        # [batch_size, hidden_size]
        charge_vecs = self.chargeLinear(hidden_state)                   # 获取charge_vecs
        # [batch_size, hidden_size]
        article_vecs = self.chargeAwareAtten(encs, charge_vecs)          # 获取article_vecs
        # [batch_size, hidden_size]
        penalty_vecs = self.articleAwareAtten(encs, article_vecs)        # 获取penalty_vecs

        # [batch_size, charge_label_size]
        charge_preds = self.chargePreds(charge_vecs)
        # [batch_size, article_label_size]
        article_preds = self.articlePreds(article_vecs)
        # [batch_size, penalty_label_size]
        penalty_preds = self.penaltyPreds(penalty_vecs)
        return charge_vecs, charge_preds, article_preds, penalty_preds

class CharacAwareAtten(nn.Module):
    def __init__(self, hidden_size, mode="concat"):
        """
        :param hidden_size: 隐变量的维度
        """
        super(CharacAwareAtten, self).__init__()
        self.hidden_size = hidden_size
        self.mode = mode
        # encoder 输出的转换矩阵
        self.transM = torch.normal(0, 1, (self.hidden_size, self.hidden_size),
                                   dtype=torch.float32,
                                   requires_grad=True)
        if mode == "sum":
            self.linear = nn.Sequential(
                nn.Linear(hidden_size, 2 * hidden_size),
                nn.BatchNorm1d(2 * hidden_size),
                nn.ReLU(),
                nn.Linear(2 * hidden_size, hidden_size)
            )
        if mode == "concat":
            self.linear = nn.Sequential(
                nn.Linear(2*hidden_size, 4 * hidden_size),
                nn.BatchNorm1d(4 * hidden_size),
                nn.ReLU(),
                nn.Linear(4 * hidden_size, hidden_size)
            )

    def forward(self, encs, vecs):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param vecs: vec  --> [batch_size, hidden_size]
        :return:chara_vecs for preds --> [batch_size, hidden_size]
        """
        # [batch_size, seq_length, hidden_size]
        encs = torch.matmul(encs, self.transM)

        # [batch_size, hidden_size, 1]
        vecs = torch.unsqueeze(vecs, dim=2)

        # [batch_size, seq_length, 1]
        atten_score = torch.matmul(encs, vecs)

        # [batch_size, seq_length, 1]
        atten_score = atten_score.softmax(dim=1)

        # [batch_size, hidden_size, seq_length]
        encs = encs.transpose(dim0=1, dim1=2)

        # [batch_size, hidden_size, 1]
        atten_sum = torch.matmul(encs, atten_score)

        # [batch_size, hidden_size]
        atten_sum = atten_sum.squeeze()

        if self.mode == "concat":
            # [batch_size, 2*hidden_size]
            chara_vecs = torch.concat((atten_sum, vecs.squeeze()), dim=1)
            # [batch_size, hidden_size]
            return self.linear(chara_vecs)
        if self.mode == "sum":
            # [batch_size, hidden_size]
            chara_vecs = atten_sum + vecs.squeeze()
            # [batch_size, hidden_size]
            return self.articleLinear(chara_vecs)

class ArticleAwareAtten(nn.Module):
    def __init__(self, hidden_size, mode="concat"):
        """
        :param hidden_size: 隐变量的维度
        """
        super(ArticleAwareAtten, self).__init__()
        # encoder 输出的转换矩阵
        self.hidden_size = hidden_size
        self.mode = mode
        self.transM = torch.normal(0, 1, (self.hidden_size, self.hidden_size),
                                   dtype=torch.float32,
                                   requires_grad=True)

    def forward(self, encs, article_vecs):
        """
        :param encs: encoder outputs --> [batch_size, seq_length, hidden_size]
        :param article_vecs: vec train by supervised contrastive learning --> [batch_size, hidden_size]
        :return:penalty_vecs for penalty preds --> [batch_size, hidden_size]
        """
        # [batch_size, seq_length, hidden_size]
        encs_transed = torch.matmul(encs, self.transM)

        # [batch_size, hidden_size, 1]
        article_vecs = torch.unsqueeze(article_vecs, dim=2)

        # [batch_size, seq_length, 1]
        atten_score = torch.matmul(encs_transed, article_vecs)

        # [batch_size, seq_length, 1]
        atten_score = atten_score.softmax(dim=1)

        # [batch_size, hidden_size, seq_length]
        encs_transed = encs_transed.transpose(dim0=1, dim1=2)

        # [batch_size, hidden_size, 1]
        atten_sum = torch.matmul(encs_transed, atten_score)

        # [batch_size, hidden_size]
        atten_sum = atten_sum.squeeze()

        if self.mode == "concat":
            penalty_vec =  torch.concat((atten_sum, article_vecs.squeeze()), dim=1)
            return
        if self.mode == "sum":
            return atten_sum + article_vecs