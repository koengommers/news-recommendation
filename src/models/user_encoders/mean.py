from typing import Optional

import torch
import torch.nn as nn


class MeanUserEncoder(nn.Module):
    def __init__(self):
        super(MeanUserEncoder, self).__init__()

    def forward(
        self, clicked_news_vector: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, embedding_dim
            mask: batch_size, num_clicked_news_a_user
        Returns:
            (shape) batch_size, embedding_dim
        """
        return clicked_news_vector.mean(dim=1)
