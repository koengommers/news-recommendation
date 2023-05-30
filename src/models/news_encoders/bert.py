from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel

from src.models.modules.attention.additive import AdditiveAttention


class BERTNewsEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        pooling_method: str = "attention",
        dropout_probability: float = 0.2,
        query_vector_dim: int = 200,
        num_hidden_layers: Optional[int] = None,
        finetune_n_last_layers: int = 2,
    ):
        super(BERTNewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability

        self.bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        if num_hidden_layers is not None:
            self.bert_config.num_hidden_layers = num_hidden_layers
        self.embedding_dim = self.bert_config.hidden_size
        self.bert_model = AutoModel.from_config(self.bert_config)

        assert pooling_method in ["attention", "average", "pooler"]
        self.pooling_method = pooling_method

        # Only finetune last layers
        if finetune_n_last_layers >= 0:
            freeze_layers = list(
                range(self.bert_config.num_hidden_layers - finetune_n_last_layers)
            )
            for name, param in self.bert_model.named_parameters():
                if (
                    name.startswith("encoder")
                    and int(name.split(".")[2]) in freeze_layers
                ):
                    param.requires_grad = False

        if pooling_method == "attention":
            self.additive_attention = AdditiveAttention(
                query_vector_dim, self.bert_config.hidden_size
            )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def stack_batches(self, news):
        batch_size, n_news, num_words = news["input_ids"].size()
        for key in news:
            news[key] = news[key].reshape(-1, num_words)

        unstack = partial(self.unstack_batches, batch_size=batch_size, n_news=n_news)

        return news, unstack

    @staticmethod
    def unstack_batches(news_vectors, batch_size, n_news):
        return news_vectors.reshape(batch_size, n_news, -1)

    def forward(self, news: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        # batch size, n news, num words
        # or
        # batch size, num words
        titles = news["title"]

        has_multiple_news = titles["input_ids"].dim() == 3
        if has_multiple_news:
            titles, unstack = self.stack_batches(titles)

        for key in titles:
            titles[key] = titles[key].to(self.device)

        bert_output = self.bert_model(**titles)
        last_hidden_state = F.dropout(
            bert_output.last_hidden_state,
            p=self.dropout_probability,
            training=self.training,
        )

        if self.pooling_method == "attention":
            news_vectors = self.additive_attention(last_hidden_state)
        elif self.pooling_method == "average":
            news_vectors = last_hidden_state.mean(dim=1)
        elif self.pooling_method == "pooler":
            news_vectors = bert_output.pooler_output
        else:
            raise ValueError("Unknown pooling method")

        if has_multiple_news:
            news_vectors = unstack(news_vectors)  # type:ignore

        return news_vectors
