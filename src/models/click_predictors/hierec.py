import torch
import torch.nn as nn
import torch.nn.functional as F


class HieRecClickPredictor(nn.Module):
    def __init__(self, lambda_t=0.15, lambda_s=0.7):
        super().__init__()
        assert (lambda_t + lambda_s) < 1
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s

    def forward(self, candidate_news, user):
        (
            subcategory_repr,
            subcategory_weights,
            category_repr,
            category_weights,
            user_repr,
        ) = user

        # Subtopic-level score
        subcategory_mask = F.one_hot(
            candidate_news["subcategory"], subcategory_repr.size(1)
        )
        subcategory_embeddings = torch.bmm(subcategory_mask.float(), subcategory_repr)
        subcategory_match = torch.sum(
            candidate_news["vectors"] * subcategory_embeddings, dim=-1
        )
        subcategory_weights = torch.bmm(
            subcategory_mask.float(), subcategory_weights.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        subcategory_score = subcategory_match * subcategory_weights

        # Topic-level score
        category_mask = F.one_hot(candidate_news["category"], category_repr.size(1))
        category_embeddings = torch.bmm(category_mask.float(), category_repr)
        category_match = torch.sum(
            candidate_news["vectors"] * category_embeddings, dim=-1
        )
        category_weights = torch.bmm(
            category_mask.float(), category_weights.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        category_score = category_match * category_weights

        # User-level score
        user_score = torch.bmm(
            candidate_news["vectors"], user_repr.unsqueeze(dim=-1)
        ).squeeze(dim=-1)
        print(user_score.size())

        # Final score
        return (
            self.lambda_s * subcategory_score
            + self.lambda_t * category_score
            + (1 - self.lambda_s - self.lambda_t) * user_score
        )
