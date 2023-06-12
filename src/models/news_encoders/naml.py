from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules.attention.additive import AdditiveAttention
from src.utils.context import context
from src.utils.data import load_pretrained_embeddings


class TextEncoder(torch.nn.Module):
    def __init__(
        self,
        word_embedding,
        num_filters,
        window_size,
        query_vector_dim,
        dropout_probability,
    ):
        super(TextEncoder, self).__init__()
        self.word_embedding = word_embedding
        self.dropout_probability = dropout_probability
        self.CNN = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding.embedding_dim),
            padding=(int((window_size - 1) / 2), 0),
        )
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, text):
        # batch_size, num_words_text, word_embedding_dim
        text_vector = F.dropout(
            self.word_embedding(text),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters, num_words_title
        convoluted_text_vector = self.CNN(text_vector.unsqueeze(dim=1)).squeeze(dim=3)
        # batch_size, num_filters, num_words_title
        activated_text_vector = F.dropout(
            F.relu(convoluted_text_vector),
            p=self.dropout_probability,
            training=self.training,
        )

        # batch_size, num_filters
        text_vector = self.additive_attention(activated_text_vector.transpose(1, 2))
        return text_vector


class ElementEncoder(torch.nn.Module):
    def __init__(self, num_elements, embedding_dim, output_dim):
        super(ElementEncoder, self).__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim, padding_idx=0)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, element):
        element_embedding = self.embedding(element)
        element_vector = self.linear(element_embedding)
        activated_element_vector = F.relu(element_vector)
        return activated_element_vector


class NAMLNewsEncoder(torch.nn.Module):
    @context.fill(
        num_words="num_words",
        token2int="token2int",
        num_categories="num_categories",
        num_subcategories="num_subcategories",
    )
    def __init__(
        self,
        num_words: int = 0,
        word_embedding_dim: int = 300,
        use_pretrained_embeddings: bool = False,
        token2int: dict[str, int] = {},
        freeze_pretrained_embeddings: bool = False,
        dropout_probability: float = 0.2,
        window_size: int = 3,
        num_filters: int = 400,
        query_vector_dim: int = 200,
        num_categories: int = 0,
        num_subcategories: int = 0,
        category_embedding_dim: int = 100,
    ):
        super(NAMLNewsEncoder, self).__init__()
        self.embedding_dim = num_filters

        if use_pretrained_embeddings:
            pretrained_embeddings = load_pretrained_embeddings(token2int)
            word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_pretrained_embeddings,
                padding_idx=0,
            )
        else:
            word_embedding = nn.Embedding(num_words, word_embedding_dim, padding_idx=0)

        self.encoders = nn.ModuleDict(
            {
                "title": TextEncoder(
                    word_embedding,
                    num_filters,
                    window_size,
                    query_vector_dim,
                    dropout_probability,
                ),
                "abstract": TextEncoder(
                    word_embedding,
                    num_filters,
                    window_size,
                    query_vector_dim,
                    dropout_probability,
                ),
                "category": ElementEncoder(
                    num_categories,
                    category_embedding_dim,
                    num_filters,
                ),
                "subcategory": ElementEncoder(
                    num_subcategories,
                    category_embedding_dim,
                    num_filters,
                ),
            }
        )
        self.final_attention = AdditiveAttention(query_vector_dim, num_filters)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def stack_batches(self, news):
        batch_size, n_news = news["category"].size()
        for key in news:
            if news[key].dim() == 3:
                news[key] = news[key].reshape(-1, news[key].size(-1))
            elif news[key].dim() == 2:
                news[key] = news[key].reshape(-1)

        unstack = partial(self.unstack_batches, batch_size=batch_size, n_news=n_news)

        return news, unstack

    @staticmethod
    def unstack_batches(news_vectors, batch_size, n_news):
        return news_vectors.reshape(batch_size, n_news, -1)

    def forward(self, news):
        has_multiple_news = news["title"].dim() == 3
        if has_multiple_news:
            news, unstack = self.stack_batches(news)

        encodings = torch.stack(
            [
                encoder(news[name].to(self.device))
                for name, encoder in self.encoders.items()
            ],
            dim=1,
        )

        final_news_vector = self.final_attention(encodings)

        if has_multiple_news:
            final_news_vector = unstack(final_news_vector)  # type:ignore

        return final_news_vector
