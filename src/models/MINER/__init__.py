from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from models.MINER.user_encoder import UserEncoder
from models.modules.bert.news_encoder import NewsEncoder


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


class MINER(nn.Module):
    """
    MINER network.
    """

    def __init__(
        self,
        pretrained_model_name: str,
        n_interest_vectors: int = 32,
        bert_pooling_method: str = "pooler",
        aggregate_method: str = "weighted",
        disagreement_loss_weight: float = 0.8,
    ):
        super(MINER, self).__init__()
        self.aggregate_method = aggregate_method
        self.disagreement_loss_weight = disagreement_loss_weight

        self.bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.news_encoder = NewsEncoder(
            self.bert_config, bert_pooling_method, finetune_n_last_layers=-1
        )
        self.user_encoder = UserEncoder(
            n_interest_vectors, word_embedding_dim=self.bert_config.hidden_size
        )
        self.loss_fn = nn.CrossEntropyLoss()

        if aggregate_method == "weighted":
            self.score_aggregator = TargetAwareAttention(self.bert_config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        candidate_news: dict[str, dict[str, torch.Tensor]],
        clicked_news: dict[str, dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_candidate_news, num_words = candidate_news["title"][
            "input_ids"
        ].size()
        for key in candidate_news["title"]:
            candidate_news["title"][key] = candidate_news["title"][key].reshape(
                -1, num_words
            )

        # batch_size, 1 + K, hidden_size
        candidate_news_vector = self.get_news_vector(candidate_news).reshape(
            batch_size, n_candidate_news, -1
        )

        # batch_size, num_clicked_news_a_user, hidden_size
        batch_size, history_length, num_words = clicked_news["title"][
            "input_ids"
        ].size()
        for key in clicked_news["title"]:
            clicked_news["title"][key] = clicked_news["title"][key].reshape(
                -1, num_words
            )

        clicked_news_vector = self.get_news_vector(clicked_news).reshape(
            batch_size, history_length, -1
        )

        # batch_size, n_interest_vectors, hidden_size
        user_vectors = self.get_user_vector(clicked_news_vector)

        dot_products = torch.bmm(user_vectors, user_vectors.transpose(1, 2))
        norms = torch.norm(user_vectors, p=2, dim=-1, keepdim=True)
        cos_similarities = dot_products / torch.bmm(norms, norms.transpose(1, 2))
        disagreement_loss = cos_similarities.mean()

        # batch_size, 1 + K, n_interest_vectors
        click_probability = self.get_prediction(candidate_news_vector, user_vectors)

        newsrec_loss = self.loss_fn(click_probability, labels)
        loss = newsrec_loss + self.disagreement_loss_weight * disagreement_loss

        return loss

    def get_news_vector(self, news: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        news_titles = news["title"]
        for key in news_titles:
            news_titles[key] = news_titles[key].to(self.device)
        return self.news_encoder(news_titles)

    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(clicked_news_vector.to(self.device))

    def get_prediction(
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
