import torch
import torch.nn as nn

from models.modules.attention.additive import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(
        self, n_interest_vectors=32, query_vector_dim=200, word_embedding_dim=768
    ):
        super(UserEncoder, self).__init__()
        self.additive_attentions = nn.ModuleList([
            AdditiveAttention(query_vector_dim, word_embedding_dim)
            for _ in range(n_interest_vectors)
        ])

    def forward(self, clicked_news_vectors):
        user_vectors = torch.stack(
            [
                additive_attention(clicked_news_vectors)
                for additive_attention in self.additive_attentions
            ], dim=1
        )
        return user_vectors