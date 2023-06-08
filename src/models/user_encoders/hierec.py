from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.context import context


class HieRecUserEncoder(nn.Module):
    @context.fill(
        num_subcategories="num_subcategories",
        num_categories="num_categories",
        news_embedding_dim="news_embedding_dim",
    )
    def __init__(
        self,
        num_subcategories=1,
        num_categories=1,
        news_embedding_dim=300,
        history_length=50,
        num_clicked_embed_dim=128,
    ):
        super().__init__()
        self.num_subcategories = num_subcategories
        self.num_categories = num_categories
        categories_embed_dim = news_embedding_dim

        self.subcategory_embedding = nn.Embedding(
            num_subcategories, categories_embed_dim
        )
        self.category_embedding = nn.Embedding(num_categories, categories_embed_dim)

        self.num_clicks_embedding = nn.Embedding(
            history_length + 1, num_clicked_embed_dim
        )
        self.num_clicks_scorer = nn.Linear(num_clicked_embed_dim, 1)

        self.news_attention_dense = nn.Linear(news_embedding_dim, 1, bias=False)
        self.subcategory_attention_dense = nn.Linear(
            categories_embed_dim, 1, bias=False
        )
        self.category_attention_dense = nn.Linear(categories_embed_dim, 1, bias=False)

    @property
    def requires_features(self):
        return True

    def forward(self, news, mask: Optional[torch.Tensor] = None):
        batch_size = news["vectors"].size(0)
        history_length = news["vectors"].size(1)
        subcategory_mask = F.one_hot(news["subcategory"], self.num_subcategories)
        category_mask = F.one_hot(news["category"], self.num_categories)
        subcategory_to_category = (
            subcategory_mask.float()
            .transpose(1, 2)
            .matmul(category_mask.float())
            .long()
        )

        # Subtopic-level interest representation
        news_attention_scores = self.news_attention_dense(news["vectors"]).squeeze(
            dim=-1
        )
        if mask is not None:
            news_attention_scores = news_attention_scores - 100 * (1 - mask)
        news_attention_scores = (
            news_attention_scores.unsqueeze(dim=-1)
            .expand(batch_size, history_length, self.num_subcategories)
            .clone()
        )
        news_attention_scores = news_attention_scores - 100 * (1 - subcategory_mask)
        news_attention_weights = F.softmax(news_attention_scores, dim=1)
        subcategory_news_repr = torch.bmm(
            news_attention_weights.transpose(1, 2), news["vectors"]
        )
        subcategory_embedding = self.subcategory_embedding(
            torch.arange(self.num_subcategories, device=subcategory_news_repr.device)
        )
        subcategory_repr = subcategory_news_repr + subcategory_embedding

        subcategory_clicks = subcategory_mask.sum(dim=1)
        subcategory_weights = subcategory_clicks / subcategory_clicks.sum(
            dim=1
        ).unsqueeze(dim=1).expand(batch_size, self.num_subcategories)

        # Topic-level interest representation
        subcategory_clicks_embedding = self.num_clicks_embedding(subcategory_clicks)
        subcategory_clicks_scores = self.num_clicks_scorer(subcategory_clicks_embedding)
        subcategory_repr_scores = self.subcategory_attention_dense(subcategory_repr)
        subcategory_attention_scores = (
            subcategory_repr_scores + subcategory_clicks_scores
        )
        subcategory_attention_scores = subcategory_attention_scores.expand(
            batch_size, self.num_subcategories, self.num_categories
        ).clone()
        subcategory_attention_scores = subcategory_attention_scores - 100 * (
            1 - subcategory_to_category
        )
        subcategory_attention_weights = F.softmax(subcategory_attention_scores, dim=1)
        category_subcategory_repr = torch.bmm(
            subcategory_attention_weights.transpose(1, 2), subcategory_repr
        )
        category_embedding = self.category_embedding(
            torch.arange(self.num_categories, device=category_subcategory_repr.device)
        )
        category_repr = category_subcategory_repr + category_embedding

        category_clicks = category_mask.sum(dim=1)
        category_weights = category_clicks / category_clicks.sum(dim=1).unsqueeze(
            dim=1
        ).expand(batch_size, self.num_categories)

        # User-level interest representation
        category_clicks_embedding = self.num_clicks_embedding(category_clicks)
        category_clicks_scores = self.num_clicks_scorer(category_clicks_embedding)
        category_repr_scores = self.category_attention_dense(category_repr)
        category_attention_scores = category_repr_scores + category_clicks_scores
        category_attention_weights = F.softmax(category_attention_scores, dim=1)
        user_repr = torch.bmm(
            category_attention_weights.transpose(1, 2), category_repr
        ).squeeze()

        return (
            subcategory_repr,
            subcategory_weights,
            category_repr,
            category_weights,
            user_repr,
        )
