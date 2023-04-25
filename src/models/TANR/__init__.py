from typing import Optional

import torch
import torch.nn as nn

from models.TANR.news_encoder import NewsEncoder
from models.TANR.user_encoder import UserEncoder


class TANR(torch.nn.Module):
    """
    TANR network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(
        self,
        num_words: int,
        num_categories: int,
        word_embedding_dim: int = 300,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_pretrained_embeddings: bool = False,
        window_size: int = 3,
        num_filters: int = 300,
        topic_classification_loss_weight: float = 0.2,
    ):
        super(TANR, self).__init__()
        self.num_filters = num_filters
        self.num_categories = num_categories
        self.topic_classification_loss_weight = topic_classification_loss_weight

        self.news_encoder = NewsEncoder(
            num_words,
            word_embedding_dim,
            pretrained_embeddings,
            freeze_pretrained_embeddings,
            window_size=window_size,
            num_filters=num_filters,
        )
        self.user_encoder = UserEncoder(num_filters=num_filters)
        self.topic_predictor = nn.Linear(num_filters, num_categories)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        candidate_news: dict[str, torch.Tensor],
        clicked_news: dict[str, torch.Tensor],
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
        batch_size, n_candidate_news, num_words = candidate_news["title"].size()
        candidate_news["title"] = candidate_news["title"].reshape(-1, num_words)
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = self.get_news_vector(candidate_news).reshape(
            batch_size, n_candidate_news, -1
        )

        batch_size, history_length, num_words = clicked_news["title"].size()
        clicked_news["title"] = clicked_news["title"].reshape(-1, num_words)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = self.get_news_vector(clicked_news).reshape(
            batch_size, history_length, -1
        )

        # batch_size, num_filters
        user_vector = self.get_user_vector(clicked_news_vector, mask)
        # batch_size, 1 + K
        click_probability = self.get_prediction(candidate_news_vector, user_vector)

        # batch_size * (1 + K + num_clicked_news_a_user), num_categories
        y_pred = self.topic_predictor(
            torch.cat((candidate_news_vector, clicked_news_vector), dim=1).view(
                -1, self.num_filters
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

    def get_news_vector(self, news: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.news_encoder(news["title"].to(self.device))

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
