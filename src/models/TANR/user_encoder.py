import torch
import torch.nn as nn

from models.modules.attention.additive import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(self, query_vector_dim: int = 200, num_filters: int = 300):
        super(UserEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, num_filters
        Returns:
            (shape) batch_size, num_filters
        """
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector
