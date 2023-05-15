from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.attention.additive import AdditiveAttention
from src.models.modules.attention.multihead_self import MultiHeadSelfAttention
from src.utils.context import context
from src.utils.data import load_pretrained_embeddings


class NRMSNewsEncoder(nn.Module):
    @context.fill(num_words="num_words", token2int="token2int")
    def __init__(
        self,
        num_words: int = 0,
        word_embedding_dim: int = 300,
        use_pretrained_embeddings: bool = False,
        token2int: dict[str, int] = {},
        freeze_pretrained_embeddings: bool = False,
        dropout_probability: float = 0.2,
        num_attention_heads: int = 15,
        query_vector_dim: int = 200,
    ):
        super(NRMSNewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.embedding_dim = word_embedding_dim

        if use_pretrained_embeddings:
            pretrained_embeddings = load_pretrained_embeddings(token2int)
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_pretrained_embeddings,
                padding_idx=0,
            )
        else:
            self.word_embedding = nn.Embedding(
                num_words, word_embedding_dim, padding_idx=0
            )

        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, word_embedding_dim
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def stack_batches(self, news):
        batch_size, n_news, num_words = news.size()
        news = news.reshape(-1, num_words)
        unstack = partial(self.unstack_batches, batch_size=batch_size, n_news=n_news)

        return news, unstack

    @staticmethod
    def unstack_batches(news_vectors, batch_size, n_news):
        return news_vectors.reshape(batch_size, n_news, -1)

    def forward(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            news: [
                {
                    "title": batch_size * num_words_title
                }
            ]
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        titles = news["title"].to(self.device)

        has_multiple_news = titles.dim() == 3
        if has_multiple_news:
            titles, unstack = self.stack_batches(titles)

        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(
            self.word_embedding(titles),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(
            multihead_news_vector,
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)

        if has_multiple_news:
            final_news_vector = unstack(final_news_vector)  # type:ignore

        return final_news_vector
