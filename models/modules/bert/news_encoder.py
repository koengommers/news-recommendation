import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from models.modules.attention.additive import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
        self,
        bert_config,
        dropout_probability=0.2,
        query_vector_dim=200,
        finetune_n_last_layers=2,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.bert_model = AutoModel.from_config(bert_config)

        # Only finetune last layers
        if finetune_n_last_layers >= 0:
            freeze_layers = list(
                range(bert_config.num_hidden_layers - finetune_n_last_layers)
            )
            for name, param in self.bert_model.named_parameters():
                if name.startswith("embedding"):
                    param.requires_grad = False
                if (
                    name.startswith("encoder")
                    and int(name.split(".")[2]) in freeze_layers
                ):
                    param.requires_grad = False

        self.additive_attention = AdditiveAttention(
            query_vector_dim, bert_config.hidden_size
        )

    def forward(self, news):
        last_hidden_state = F.dropout(
            self.bert_model(**news).last_hidden_state,
            p=self.dropout_probability,
            training=self.training,
        )
        news_vector = self.additive_attention(last_hidden_state)
        return news_vector