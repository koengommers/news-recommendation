from typing import Optional

import torch
import torch.nn as nn

from models.NRMS.news_encoder import NewsEncoder
from models.NRMS.user_encoder import UserEncoder


class NRMS(nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(
        self,
        num_words: int,
        word_embedding_dim: int = 300,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_pretrained_embeddings: bool = False,
    ):
        super(NRMS, self).__init__()
        self.news_encoder = NewsEncoder(
            num_words,
            word_embedding_dim,
            pretrained_embeddings,
            freeze_pretrained_embeddings,
        )
        self.user_encoder = UserEncoder(word_embedding_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        candidate_news: dict[str, torch.Tensor],
        clicked_news: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        candidate_news_titles = candidate_news["title"].to(self.device)
        clicked_news_titles = clicked_news["title"].to(self.device)

        batch_size, n_candidate_news, num_words = candidate_news_titles.size()
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = self.news_encoder(
            candidate_news_titles.reshape(-1, num_words)
        ).reshape(batch_size, n_candidate_news, -1)

        batch_size, history_length, num_words = clicked_news_titles.size()
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = self.news_encoder(
            clicked_news_titles.reshape(-1, num_words)
        ).reshape(batch_size, history_length, -1)

        # batch_size, word_embedding_dim
        user_vector = self.user_encoder(clicked_news_vector)

        # batch_size, 1 + K
        click_probability = torch.bmm(
            candidate_news_vector, user_vector.unsqueeze(dim=-1)
        ).squeeze(dim=-1)

        return self.loss_fn(click_probability, labels)

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

    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector.to(self.device))

    def get_prediction(self, news_vector: torch.Tensor, user_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        news_vector = news_vector.unsqueeze(0)
        user_vector = user_vector.unsqueeze(0)
        probability = (
            torch.bmm(news_vector, user_vector.unsqueeze(dim=-1))
            .squeeze(dim=-1)
            .squeeze(dim=0)
        )
        return probability
