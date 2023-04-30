import torch
import torch.nn as nn

from utils.context import context


class TopicPredictionLoss(nn.Module):
    def __init__(
        self,
        weight=0.2,
        news_embedding_dim=context.read("news_embedding_dim"),
        num_categories=context.read("num_categories"),
    ):
        super(TopicPredictionLoss, self).__init__()
        self.weight = weight
        self.news_embedding_dim = news_embedding_dim
        self.topic_predictor = nn.Linear(news_embedding_dim, num_categories)
        class_weight = torch.ones(num_categories).to(self.device)
        class_weight[0] = 0
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, values):
        candidate_news_vectors = values["candidate_news_vectors"]
        clicked_news_vectors = values["clicked_news_vectors"]
        candidate_news = values["candidate_news"]
        clicked_news = values["clicked_news"]
        # batch_size * (1 + K + num_clicked_news_a_user), num_categories
        y_pred = self.topic_predictor(
            torch.cat((candidate_news_vectors, clicked_news_vectors), dim=1).view(
                -1, self.news_embedding_dim
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

        return self.weight * self.loss_fn(y_pred, y)
