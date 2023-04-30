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
        dot_products = torch.bmm(user_vectors, user_vectors.transpose(1, 2))
        norms = torch.norm(user_vectors, p=2, dim=-1, keepdim=True)
        cos_similarities = dot_products / torch.bmm(norms, norms.transpose(1, 2))
        disagreement_loss = cos_similarities.mean()
        return self.weight * disagreement_loss
