import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention.additive import AdditiveAttention
from models.modules.attention.multihead_self import MultiHeadSelfAttention
from utils.data import load_pretrained_embeddings


class NewsEncoder(nn.Module):
    def __init__(
        self,
        dataset,
        word_embedding_dim: int = 300,
        use_pretrained_embeddings: bool = False,
        freeze_pretrained_embeddings: bool = False,
        dropout_probability: float = 0.2,
        num_attention_heads: int = 15,
        query_vector_dim: int = 200,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.embedding_dim = 300

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

        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, word_embedding_dim
        )

    def forward(self, news: torch.Tensor) -> torch.Tensor:
        """
        Args:
            news: batch_size * num_words_title
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        news_vector = F.dropout(
            self.word_embedding(news),
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
        return final_news_vector
