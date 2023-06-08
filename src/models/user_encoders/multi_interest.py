from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.context import context
from src.utils.utils import masked_softmax


class PolyAttention(nn.Module):
    def __init__(self, embed_dim, n_context_codes, context_code_dim):
        super().__init__()
        self.projection = nn.Linear(embed_dim, context_code_dim)
        self.context_codes = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(context_code_dim, n_context_codes),
                gain=nn.init.calculate_gain("tanh"),
            )
        )

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        proj = torch.tanh(self.projection(x))
        attn_scores = torch.matmul(proj, self.context_codes)
        attn_scores = attn_scores.transpose(1, 2)
        if attn_mask is not None:
            # batch_size, candidate_size
            attn_weights = masked_softmax(attn_scores, attn_mask.unsqueeze(1), dim=2)
        else:
            attn_weights = F.softmax(attn_scores, dim=2)
        poly_repr = torch.bmm(attn_weights, x)

        return poly_repr


class MultiInterestUserEncoder(nn.Module):
    @context.fill(news_embedding_dim="news_embedding_dim")
    def __init__(
        self,
        news_embedding_dim: int = 300,
        n_interest_vectors: int = 32,
        query_vector_dim: int = 200,
    ) -> None:
        super(MultiInterestUserEncoder, self).__init__()
        self.poly_attention = PolyAttention(
            news_embedding_dim, n_interest_vectors, query_vector_dim
        )

    def forward(
        self, clicked_news_vectors: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.poly_attention(clicked_news_vectors, mask)
