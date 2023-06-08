from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.utils import masked_softmax


class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """

    def __init__(
        self,
        query_vector_dim: int,
        embed_dim: int,
    ):
        super(AdditiveAttention, self).__init__()
        self.projection = nn.Linear(embed_dim, query_vector_dim)
        self.query_vector = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.empty(query_vector_dim, 1), gain=nn.init.calculate_gain("tanh")
            ).squeeze()
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, query_vector_dim
        proj = torch.tanh(self.projection(x))
        attn_scores = torch.matmul(proj, self.query_vector)

        if mask is not None:
            # batch_size, candidate_size
            attn_weights = masked_softmax(attn_scores, mask, dim=1)
        else:
            attn_weights = F.softmax(attn_scores, dim=1)

        # batch_size, candidate_vector_dim
        repr = torch.bmm(attn_weights.unsqueeze(dim=1), x).squeeze(dim=1)
        return repr
