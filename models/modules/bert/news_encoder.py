import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.modules.attention.additive import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(self, bert_config, dropout_probability=0.2, query_vector_dim=200):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.bert_model = AutoModel.from_config(bert_config)
        self.additive_attention = AdditiveAttention(
            query_vector_dim,
            bert_config.hidden_size
        )

    def forward(self, news):
        last_hidden_state = F.dropout(
            self.bert_model(**news).last_hidden_state,
            p=self.dropout_probability,
            training=self.training,
        )
        news_vector = self.additive_attention(last_hidden_state)
        return news_vector
