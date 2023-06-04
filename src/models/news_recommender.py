import torch
import torch.nn as nn


class NewsRecommender(nn.Module):
    def __init__(self, news_encoder, user_encoder, click_predictor, loss_modules):
        super(NewsRecommender, self).__init__()
        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.click_predictor = click_predictor
        self.loss_modules = nn.ModuleList(loss_modules)

        self.pass_features = (
            hasattr(self.user_encoder, "requires_features")
            and self.user_encoder.requires_features
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_news(self, news):
        return self.news_encoder(news)

    def encode_user(self, clicked_news, mask=None):
        clicked_news = clicked_news.to(self.device)
        if mask is not None:
            return self.user_encoder(clicked_news, mask.to(self.device))
        return self.user_encoder(clicked_news)

    def rank(self, candidate_news, user_vector):
        return self.click_predictor(candidate_news, user_vector)

    def forward(self, candidate_news, clicked_news, labels, mask=None):
        candidate_news_vectors = self.encode_news(candidate_news)
        clicked_news_vectors = self.encode_news(clicked_news)

        if self.pass_features:
            candidate_news_repr = candidate_news
            candidate_news_repr["vectors"] = candidate_news_vectors
            clicked_news_repr = clicked_news
            clicked_news_repr["vectors"] = clicked_news_vectors
        else:
            candidate_news_repr = candidate_news_vectors
            clicked_news_repr = clicked_news_vectors

        user_vector = self.encode_user(clicked_news_repr, mask)
        click_probability = self.rank(candidate_news_repr, user_vector)

        loss = sum(
            [
                loss_module(
                    {
                        "candidate_news": candidate_news,
                        "clicked_news": clicked_news,
                        "candidate_news_vectors": candidate_news_vectors,
                        "clicked_news_vectors": clicked_news_vectors,
                        "user_vector": user_vector,
                        "click_probability": click_probability,
                        "labels": labels,
                    }
                )
                for loss_module in self.loss_modules
            ]
        )

        return loss
