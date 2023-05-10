import torch
import torch.nn as nn


class InterestDisagreementLoss(nn.Module):
    def __init__(self, weight=0.8):
        super(InterestDisagreementLoss, self).__init__()

        self.weight = weight

    def forward(
        self,
        values,
    ):
        user_vectors = values["user_vector"]

        batch_size, n_interest_vectors, _ = user_vectors.size()
        n_pairs = (n_interest_vectors * (n_interest_vectors - 1)) / 2

        dot_products = torch.bmm(user_vectors, user_vectors.transpose(1, 2))
        norms = torch.norm(user_vectors, p=2, dim=-1, keepdim=True)
        cos_similarities = dot_products / torch.bmm(norms, norms.transpose(1, 2))

        upper_right = cos_similarities.triu(diagonal=1)
        disagreement_loss = upper_right.sum() / (batch_size * n_pairs)

        return self.weight * disagreement_loss
