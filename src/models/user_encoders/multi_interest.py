from typing import Optional

import torch
import torch.nn as nn

from models.modules.attention.additive import AdditiveAttention
from utils.context import context


class MultiInterestUserEncoder(nn.Module):
    def __init__(
        self,
        news_embedding_dim: int = context.read("news_embedding_dim", default=300),
        n_interest_vectors: int = 32,
        query_vector_dim: int = 200,
    ) -> None:
        super(MultiInterestUserEncoder, self).__init__()
        self.additive_attentions = nn.ModuleList(
            [
                AdditiveAttention(query_vector_dim, news_embedding_dim)
                for _ in range(n_interest_vectors)
            ]
        )

    def forward(
        self, clicked_news_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        user_vectors = torch.stack(
            [
                additive_attention(clicked_news_vectors, mask)
                for additive_attention in self.additive_attentions
            ],
            dim=1,
        )
        return user_vectors
