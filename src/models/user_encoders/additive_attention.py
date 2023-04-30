from typing import Optional

import torch
import torch.nn as nn

from models.modules.attention.additive import AdditiveAttention
from utils.context import context


class AdditiveAttentionUserEncoder(nn.Module):
    def __init__(
        self,
        query_vector_dim: int = 200,
        num_filters: int = context.read("news_embedding_dim", default=300),
    ):
        super(AdditiveAttentionUserEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(
        self, clicked_news_vector: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(clicked_news_vector, mask)
        return user_vector
