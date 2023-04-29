from typing import Optional

import torch
import torch.nn as nn

from models.TANR.user_encoder import UserEncoder
from utils.context import context


class TANR(torch.nn.Module):
    """
    TANR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(
        self,
        news_encoder,
        num_categories: int = context.read("num_categories", default=0),
        topic_classification_loss_weight: float = 0.2,
    ):
        super(TANR, self).__init__()
        self.topic_classification_loss_weight = topic_classification_loss_weight

        self.news_encoder = news_encoder
        self.user_encoder = UserEncoder()
        self.num_categories = num_categories
        self.topic_predictor = nn.Linear(
            news_encoder.embedding_dim, self.num_categories
        )
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        candidate_news,
        clicked_news,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            candidate_news:
                [
                    {
                        "category": batch_size,
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "category": batch_size,
                        "title": batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
            click_probability: batch_size, 1 + K
            topic_classification_loss: 0-dim tensor
        """
        # batch_size, 1 + K, hidden_size
        candidate_news_vector = self.get_news_vector(candidate_news)

        # batch_size, num_clicked_news_a_user, hidden_size
        clicked_news_vector = self.get_news_vector(clicked_news)

        # batch_size, hidden_size
        user_vector = self.get_user_vector(clicked_news_vector, mask)

        # batch_size, 1 + K
        click_probability = self.get_prediction(candidate_news_vector, user_vector)

        # batch_size * (1 + K + num_clicked_news_a_user), num_categories
        y_pred = self.topic_predictor(
            torch.cat((candidate_news_vector, clicked_news_vector), dim=1).view(
                -1, self.news_encoder.embedding_dim
            )
        )
        # batch_size * (1 + K + num_clicked_news_a_user)
        y = (
            torch.cat(
                (
                    candidate_news["category"].reshape(-1),
                    clicked_news["category"].reshape(-1),
                )
            )
        ).to(self.device)
        class_weight = torch.ones(self.num_categories).to(self.device)
        class_weight[0] = 0
        criterion = nn.CrossEntropyLoss(weight=class_weight)
        topic_classification_loss = criterion(y_pred, y)
        newsrec_loss = self.loss_fn(click_probability, labels)
        loss = (
            newsrec_loss
            + self.topic_classification_loss_weight * topic_classification_loss
        )

        return loss

    def get_news_vector(self, news) -> torch.Tensor:
        # batch_size, embedding_dim
        return self.news_encoder(news)

    def get_user_vector(
        self, clicked_news_vector: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
            mask: batch_size, num_clicked_news_a_user
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        if mask is not None:
            return self.user_encoder(
                clicked_news_vector.to(self.device), mask.to(self.device)
            )
        else:
            return self.user_encoder(clicked_news_vector.to(self.device))

    def get_prediction(
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
