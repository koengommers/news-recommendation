import torch

from models.modules.attention.additive import AdditiveAttention
from models.modules.attention.multihead_self import MultiHeadSelfAttention


class UserEncoder(torch.nn.Module):
    def __init__(
        self, word_embedding_dim, query_vector_dim=200, num_attention_heads=15
    ):
        super(UserEncoder, self).__init__()
        self.multihead_self_attention = MultiHeadSelfAttention(
            word_embedding_dim, num_attention_heads
        )
        self.additive_attention = AdditiveAttention(
            query_vector_dim, word_embedding_dim
        )

    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(user_vector)
        # batch_size, word_embedding_dim
        final_user_vector = self.additive_attention(multihead_user_vector)
        return final_user_vector
