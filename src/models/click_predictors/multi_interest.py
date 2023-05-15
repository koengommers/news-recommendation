import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.context import context


class TargetAwareAttention(nn.Module):
    def __init__(self, dimension: int):
        super(TargetAwareAttention, self).__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(
        self,
        interest_vectors: torch.Tensor,
        candidate_news_vectors: torch.Tensor,
        matching_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            interest_vectors: batch_size, n_interest_vectors, hidden_dim
            candidate_news_vectors: batch_size, candidate_size, hidden_dim
            matching_scores: batch_size, candidate_size, n_interest_vectors
        Returns:
            (shape) batch_size, candidate_size
        """
        # batch_size, candidate_size, hidden_dim
        temp = F.gelu(self.linear(candidate_news_vectors))

        # batch_size, candidate_size, n_interest_vectors
        weights = F.softmax(torch.bmm(temp, interest_vectors.transpose(1, 2)), dim=2)

        # batch_size, candidate_size
        scores = torch.mul(weights, matching_scores).sum(dim=2)
        return scores


class MultiInterestClickPredictor(nn.Module):
    @context.fill(news_embedding_dim="news_embedding_dim")
    def __init__(
        self,
        news_embedding_dim: int = 300,
        aggregate_method: str = "weighted",
    ):
        super(MultiInterestClickPredictor, self).__init__()
        self.aggregate_method = aggregate_method

        if aggregate_method == "weighted":
            self.score_aggregator = TargetAwareAttention(news_embedding_dim)

    def forward(
        self, news_vector: torch.Tensor, user_vectors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            news_vector: batch_size, candidate_size, hidden_size
            user_vector: batch_size, hidden_size
        Returns:
            click_probability: batch_size, candidate_size
        """
        # batch_size, 1 + K, n_interest_vectors
        matching_scores = torch.bmm(news_vector, user_vectors.transpose(1, 2))

        # batch_size, 1 + K
        if self.aggregate_method == "max":
            click_probability = torch.max(matching_scores, dim=2)[0]
        elif self.aggregate_method == "average":
            click_probability = torch.mean(matching_scores, dim=2)
        elif self.aggregate_method == "weighted":
            click_probability = self.score_aggregator(
                user_vectors, news_vector, matching_scores
            )
        else:
            raise ValueError("Unknown aggregate method")

        return click_probability
