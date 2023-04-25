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
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        # batch_size, word_embedding_dim
        user_vector = self.get_user_vector(clicked_news_vector, mask)

        # batch_size, 1 + K
        click_probability = self.get_prediction(candidate_news_vector, user_vector)

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
