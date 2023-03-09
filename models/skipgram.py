import torch.nn as nn
import torch.nn.functional as F


class Skipgram(nn.Module):
    def __init__(
        self, n_topics, n_users, embedding_dim=100, hidden_layer_dim=128, context_size=1
    ):
        super(Skipgram, self).__init__()
        self.embeddings = nn.Embedding(n_topics, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim, hidden_layer_dim)
        self.output_layer = nn.Linear(hidden_layer_dim, context_size * n_users)
        self.context_size = context_size

    def forward(self, x):
        batch_size = x.size(0)

        embedding = self.embeddings(x)
        hidden = F.relu(self.hidden_layer(embedding))
        out = self.output_layer(hidden)
        probs = F.log_softmax(out, dim=1)
        return probs.view(batch_size, -1, self.context_size)
