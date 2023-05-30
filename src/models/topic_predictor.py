import torch.nn as nn


class TopicPredictor(nn.Module):
    def __init__(self, news_encoder, num_categories):
        super().__init__()
        news_embeddings_dim = news_encoder.embedding_dim

        self.news_encoder = news_encoder
        self.linear = nn.Linear(news_embeddings_dim, num_categories)

    def forward(self, news):
        news_vectors = self.news_encoder(news)
        predictions = self.linear(news_vectors)
        return predictions
