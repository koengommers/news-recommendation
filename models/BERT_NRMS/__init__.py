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

    def __init__(self, pretrained_model_name, bert_pooling_method="attention"):
        super(BERT_NRMS, self).__init__()
        bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.news_encoder = NewsEncoder(bert_config, bert_pooling_method)
        self.user_encoder = UserEncoder(bert_config.hidden_size, num_attention_heads=16)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, candidate_news, clicked_news, labels):
        device = next(self.parameters()).device

        batch_size, n_candidate_news, num_words = candidate_news["input_ids"].size()
        for key in candidate_news:
            candidate_news[key] = candidate_news[key].reshape(-1, num_words).to(device)

        candidate_news_vector = self.news_encoder(candidate_news).reshape(
            batch_size, n_candidate_news, -1
        )

        batch_size, history_length, num_words = clicked_news["input_ids"].size()
        for key in clicked_news:
            clicked_news[key] = clicked_news[key].reshape(-1, num_words).to(device)

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
