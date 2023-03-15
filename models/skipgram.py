import torch.nn as nn


class Skipgram(nn.Module):
    def __init__(
        self, n_topics, n_users, embedding_dim=100
    ):
        super(Skipgram, self).__init__()
        self.target_embeddings = nn.Embedding(n_topics, embedding_dim)
        self.context_embeddings = nn.Embedding(n_users, embedding_dim)

    def forward(self, target, context):
        target_embedding = self.target_embeddings(target)
        batch_size, embedding_dim = target_embedding.size()
        target_embedding = target_embedding.view(batch_size, 1, embedding_dim)

        context_embedding = self.context_embeddings(context)
        context_embedding = context_embedding.transpose(1, 2)

        sample_size = context.size(1)
        dots = target_embedding.bmm(context_embedding)
        dots = dots.view(batch_size, sample_size)

        return dots
