import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class ContrasBert(nn.Module):
    def __init__(self, hidden_size, label_size):
        super(ContrasBert, self).__init__()
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

        self.classifyLinear = nn.Sequential(
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
        contra_hidden = self.contraLinear(hidden_state)
        # [batch_size, hidden_size] -> [batch_size, label_size]
        classify_preds = self.classifyLinear(contra_hidden)
        # [batch_size, hidden_size]、[batch_size, label_size]
        return contra_hidden, classify_preds