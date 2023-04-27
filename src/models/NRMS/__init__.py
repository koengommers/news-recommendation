from typing import Optional

import torch
import torch.nn as nn

from models.NRMS.user_encoder import UserEncoder


class NRMS(nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(
        self,
        news_encoder,
        dataset,
        num_attention_heads: int = 15,
    ):
        super(NRMS, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = UserEncoder(
            news_encoder.embedding_dim, num_attention_heads=num_attention_heads
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
        # batch_size, 1 + K, word_embedding_dim
        candidate_news_vector = self.get_news_vector(candidate_news)

        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = self.get_news_vector(clicked_news)

        # batch_size, word_embedding_dim
        user_vector = self.get_user_vector(clicked_news_vector, mask)

        # batch_size, 1 + K
        click_probability = self.get_prediction(candidate_news_vector, user_vector)
        return self.loss_fn(click_probability, labels)

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
