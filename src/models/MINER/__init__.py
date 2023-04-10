import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig

from models.MINER.user_encoder import UserEncoder
from models.modules.bert.news_encoder import NewsEncoder


class TargetAwareAttention(nn.Module):
    def __init__(self, dimension):
        super(TargetAwareAttention, self).__init__()
        self.linear = nn.Linear(dimension, dimension)

    def forward(self, interest_vectors, candidate_news_vectors, matching_scores):
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
        pretrained_model_name,
        n_interest_vectors=32,
        bert_pooling_method="pooler",
        aggregate_method="weighted",
        disagreement_loss_weight=0.8,
    ):
        super(MINER, self).__init__()
        self.aggregate_method = aggregate_method
        self.disagreement_loss_weight = disagreement_loss_weight

        bert_config = AutoConfig.from_pretrained(pretrained_model_name)
        self.news_encoder = NewsEncoder(
            bert_config, bert_pooling_method, finetune_n_last_layers=-1
        )
        self.user_encoder = UserEncoder(
            n_interest_vectors, word_embedding_dim=bert_config.hidden_size
        )
        self.loss_fn = nn.CrossEntropyLoss()

        if aggregate_method == "weighted":
            self.score_aggregator = TargetAwareAttention(bert_config.hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, candidate_news, clicked_news, labels):
        candidate_news = candidate_news["title"]
        clicked_news = clicked_news["title"]

        batch_size, n_candidate_news, num_words = candidate_news["input_ids"].size()
        for key in candidate_news:
            candidate_news[key] = candidate_news[key].reshape(-1, num_words).to(self.device)
        # batch_size, n_candidates, hidden_size
        candidate_news_vector = self.news_encoder(candidate_news).reshape(
            batch_size, n_candidate_news, -1
        )

        batch_size, history_length, num_words = clicked_news["input_ids"].size()
        for key in clicked_news:
            clicked_news[key] = clicked_news[key].reshape(-1, num_words).to(self.device)
        # batch_size, history_length, hidden_size
        clicked_news_vector = self.news_encoder(clicked_news).reshape(
            batch_size, history_length, -1
        )

        # batch_size, n_interest_vectors, hidden_size
        user_vectors = self.user_encoder(clicked_news_vector)

        dot_products = torch.bmm(user_vectors, user_vectors.transpose(1, 2))
        norms = torch.norm(user_vectors, p=2, dim=-1, keepdim=True)
        cos_similarities = dot_products / torch.bmm(norms, norms.transpose(1, 2))
        disagreement_loss = cos_similarities.mean()

        # batch_size, 1 + K, n_interest_vectors
        matching_scores = torch.bmm(candidate_news_vector, user_vectors.transpose(1, 2))

        # batch_size, 1 + K
        if self.aggregate_method == "max":
            click_probability = torch.max(matching_scores, dim=2)[0]
        elif self.aggregate_method == "average":
            click_probability = torch.mean(matching_scores, dim=2)
        elif self.aggregate_method == "weighted":
            click_probability = self.score_aggregator(
                user_vectors, candidate_news_vector, matching_scores
            )

        newsrec_loss = self.loss_fn(click_probability, labels)
        loss = newsrec_loss + self.disagreement_loss_weight * disagreement_loss

        return loss

    def get_news_vector(self, news):
        news = news["title"]
        for key in news:
            news[key] = news[key].to(self.device)
        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        return self.user_encoder(clicked_news_vector.to(self.device))

    def get_prediction(self, news_vector, user_vectors):
        """
        Args:
            news_vector: candidate_size, word_embedding_dim
            user_vector: word_embedding_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        news_vector = news_vector.unsqueeze(0)
        user_vectors = user_vectors.unsqueeze(0)
        matching_scores = torch.bmm(news_vector, user_vectors.transpose(1, 2))

        # batch_size, 1 + K
        if self.aggregate_method == "max":
            click_probability = torch.max(matching_scores, dim=2)[0].squeeze(0)
        elif self.aggregate_method == "average":
            click_probability = torch.mean(matching_scores, dim=2).squeeze(0)
        elif self.aggregate_method == "weighted":
            click_probability = self.score_aggregator(
                user_vectors, news_vector, matching_scores
            ).squeeze(0)

        return click_probability
