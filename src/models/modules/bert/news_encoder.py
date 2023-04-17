from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig

from models.modules.attention.additive import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
        self,
        bert_config: PretrainedConfig,
        pooling_method: str = "attention",
        dropout_probability: float = 0.2,
        query_vector_dim: int = 200,
        num_hidden_layers: Optional[int] = None,
        finetune_n_last_layers: int = 2,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        if num_hidden_layers is not None:
            bert_config.num_hidden_layers = num_hidden_layers
        self.bert_model = AutoModel.from_config(bert_config)

        assert pooling_method in ["attention", "average", "pooler"]
        self.pooling_method = pooling_method

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

        if pooling_method == "attention":
            self.additive_attention = AdditiveAttention(
                query_vector_dim, bert_config.hidden_size
            )

    def forward(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        bert_output = self.bert_model(**news)
        last_hidden_state = F.dropout(
            bert_output.last_hidden_state,
            p=self.dropout_probability,
            training=self.training,
        )

        if self.pooling_method == "attention":
            return self.additive_attention(last_hidden_state)
        elif self.pooling_method == "average":
            return last_hidden_state.mean(dim=1)
        elif self.pooling_method == "pooler":
            return bert_output.pooler_output
        else:
            raise ValueError("Unknown pooling method")
