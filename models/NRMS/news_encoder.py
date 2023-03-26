import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention.additive import AdditiveAttention
from models.modules.attention.multihead_self import MultiHeadSelfAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NewsEncoder(torch.nn.Module):
    def __init__(
        self,
        num_words,
        word_embedding_dim=300,
        num_attention_heads=15,
        query_vector_dim=200,
        dropout_probability=0.2,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability
        self.word_embedding = nn.Embedding(num_words, word_embedding_dim, padding_idx=0)
        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, word_embedding_dim
        )

    def forward(self, news):
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
