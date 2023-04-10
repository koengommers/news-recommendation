import torch.nn as nn
import torch.nn.functional as F

from models.modules.attention.additive import AdditiveAttention


class NewsEncoder(nn.Module):
    def __init__(
        self,
        num_words,
        word_embedding_dim=300,
        pretrained_embeddings=None,
        freeze_pretrained_embeddings=False,
        dropout_probability=0.2,
        window_size=3,
        num_filters=300,
        query_vector_dim=200,
    ):
        super(NewsEncoder, self).__init__()
        self.dropout_probability = dropout_probability

        if pretrained_embeddings is not None:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_pretrained_embeddings,
                padding_idx=0,
            )
        else:
            self.word_embedding = nn.Embedding(
                num_words, word_embedding_dim, padding_idx=0
            )

        assert window_size >= 1 and window_size % 2 == 1
        self.title_CNN = nn.Conv2d(
            1,
            num_filters,
            (window_size, word_embedding_dim),
            padding=((window_size - 1) // 2, 0),
        )
        self.title_attention = AdditiveAttention(query_vector_dim, num_filters)

    def forward(self, news):
        """
        Args:
            news: batch_size * num_words_title
        Returns:
            (shape) batch_size, num_filters
        """
        # batch_size, num_words_title, word_embedding_dim
        title_vector = F.dropout(
            self.word_embedding(news),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters, num_words_title
        convoluted_title_vector = self.title_CNN(title_vector.unsqueeze(dim=1)).squeeze(
            dim=3
        )
        # batch_size, num_filters, num_words_title
        activated_title_vector = F.dropout(
            F.relu(convoluted_title_vector),
            p=self.dropout_probability,
            training=self.training,
        )
        # batch_size, num_filters
        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2)
        )

        return weighted_title_vector