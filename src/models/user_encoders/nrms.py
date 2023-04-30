from typing import Optional

import torch

from models.modules.attention.additive import AdditiveAttention
from models.modules.attention.multihead_self import MultiHeadSelfAttention
from utils.context import context


class NRMSUserEncoder(torch.nn.Module):
    def __init__(
        self,
        news_embedding_dim: int = context.read("news_embedding_dim", default=300),
        query_vector_dim: int = 200,
        num_attention_heads: int = 15,
    ):
        super(NRMSUserEncoder, self).__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(
            news_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, news_embedding_dim
        )

    def forward(
        self, clicked_news_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(
            clicked_news_vectors, mask=mask
        )
        # batch_size, word_embedding_dim
        user_vector = self.additive_attention(multihead_user_vector, mask)
        return user_vector
