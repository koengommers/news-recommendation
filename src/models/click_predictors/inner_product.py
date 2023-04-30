import torch
import torch.nn as nn


class InnerProductClickPredictor(nn.Module):
    def __init__(self):
        super(InnerProductClickPredictor, self).__init__()

    def forward(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            news_vector: batch_size, candidate_size, word_embedding_dim
            user_vector: batch_size, word_embedding_dim
        Returns:
            click_probability: batch_size, candidate_size
        """
        probability = torch.bmm(news_vector, user_vector.unsqueeze(dim=-1)).squeeze(
            dim=-1
        )
        return probability
