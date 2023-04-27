import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention.additive import AdditiveAttention
from utils.data import load_pretrained_embeddings


class NewsEncoder(nn.Module):
    def __init__(
        self,
        dataset,
        word_embedding_dim: int = 300,
        use_pretrained_embeddings: bool = False,
        freeze_pretrained_embeddings: bool = False,
        dropout_probability: float = 0.2,
        window_size: int = 3,
        num_filters: int = 300,
        query_vector_dim: int = 200,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.embedding_dim = num_filters

        if use_pretrained_embeddings:
            pretrained_embeddings = load_pretrained_embeddings(dataset.tokenizer.t2i)
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_pretrained_embeddings,
                padding_idx=0,
            )
        else:
            self.word_embedding = nn.Embedding(
                dataset.num_words, word_embedding_dim, padding_idx=0
            )

        assert window_size >= 1 and window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=((window_size - 1) // 2, 0),
        )
        self.title_attention = AdditiveAttention(query_vector_dim, num_filters)

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
            (shape) batch_size, num_filters
        """
        titles = news["title"].to(self.device)

        has_multiple_news = titles.dim() == 3
        if has_multiple_news:
            titles, unstack = self.stack_batches(titles)

        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(
            self.word_embedding(titles),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(title_vector.unsqueeze(dim=1)).squeeze(
            dim=3
        )
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(
            F.relu(convoluted_title_vector),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2)
        )

        if has_multiple_news:
            weighted_title_vector = unstack(weighted_title_vector)  # type:ignore

        return weighted_title_vector
