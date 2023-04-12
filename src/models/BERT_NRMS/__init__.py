import torch
import torch.nn as nn
from transformers import AutoConfig

from models.modules.bert.news_encoder import NewsEncoder
from models.NRMS.user_encoder import UserEncoder


class BERT_NRMS(nn.Module):
    """
    BERT-NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """

    def __init__(
        self, pretrained_model_name: str, bert_pooling_method: str = "attention"
    ):
        super(BERT_NRMS, self).__init__()
        bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.news_encoder = NewsEncoder(bert_config, bert_pooling_method)
        self.user_encoder = UserEncoder(bert_config.hidden_size, num_attention_heads=16)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        candidate_news: dict[str, dict[str, torch.Tensor]],
        clicked_news: dict[str, dict[str, torch.Tensor]],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        candidate_news_titles = candidate_news["title"]
        clicked_news_titles = clicked_news["title"]

        batch_size, n_candidate_news, num_words = candidate_news_titles[
            "input_ids"
        ].size()
        for key in candidate_news_titles:
            candidate_news_titles[key] = (
                candidate_news_titles[key].reshape(-1, num_words).to(self.device)
            )

        candidate_news_vector = self.news_encoder(candidate_news_titles).reshape(
            batch_size, n_candidate_news, -1
        )

        batch_size, history_length, num_words = clicked_news_titles["input_ids"].size()
        for key in clicked_news_titles:
            clicked_news_titles[key] = (
                clicked_news_titles[key].reshape(-1, num_words).to(self.device)
            )

        clicked_news_vector = self.news_encoder(clicked_news).reshape(
            batch_size, history_length, -1
        )

        # batch_size, hidden_size
        user_vector = self.user_encoder(clicked_news_vector)

        # batch_size, 1 + K
        click_probability = torch.bmm(
            candidate_news_vector, user_vector.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        return self.loss_fn(click_probability, labels)

    def get_news_vector(self, news: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        titles = news["title"]
        for key in titles:
            titles[key] = titles[key].to(self.device)
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(clicked_news_vector.to(self.device))

    def get_prediction(
        self, news_vector: torch.Tensor, user_vector: torch.Tensor
    ) -> torch.Tensor:
        news_vector = news_vector.unsqueeze(0)
        user_vector = user_vector.unsqueeze(0)
        probability = (
            torch.bmm(news_vector, user_vector.unsqueeze(dim=-1))
            .squeeze(dim=-1)
            .squeeze(dim=0)
        )
        return probability
