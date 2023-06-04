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
        categories_embed_dim=400,
        history_length=50,
        num_clicked_embed_dim=50,
    ):
        super().__init__()
        self.num_subcategories = num_subcategories
        self.num_categories = num_categories

        self.subcategory_embedding = nn.Embedding(
            num_subcategories, categories_embed_dim
        )
        self.category_embedding = nn.Embedding(num_categories, categories_embed_dim)

        self.num_clicks_embedding = nn.Embedding(history_length, num_clicked_embed_dim)

        self.news_attention_dense = nn.Linear(news_embedding_dim, 1, bias=False)
        self.subcategory_attention_dense = nn.Linear(
            categories_embed_dim + num_clicked_embed_dim, 1, bias=False
        )
        self.category_attention_dense = nn.Linear(
            categories_embed_dim + num_clicked_embed_dim, 1, bias=False
        )

    @property
    def requires_features(self):
        return True

    def forward(self, news, mask: Optional[torch.Tensor] = None):
        batch_size = news["vectors"].size(0)
        history_length = news["vectors"].size(1)
        subcategory_mask = F.one_hot(news["subcategory"], self.num_subcategories)
        category_mask = F.one_hot(news["category"], self.num_categories)
        subcategory_to_category = subcategory_mask.transpose(1, 2).matmul(category_mask)

        # Subtopic-level interest representation
        news_attention_scores = self.news_attention_dense(news["vectors"])
        if mask:
            news_attention_scores.masked_fill_(~mask.bool(), 1e-30)
        news_attention_scores = news_attention_scores.expand(
            batch_size, history_length, self.num_subcategories
        ).clone()
        news_attention_scores.masked_fill_(~subcategory_mask.bool(), 1e-30)
        news_attention_weights = F.softmax(news_attention_scores, dim=1)
        subcategory_news_repr = torch.bmm(
            news_attention_weights.transpose(1, 2), news["vectors"]
        )
        subcategory_embedding = self.subcategory_embedding(
            torch.arange(self.num_subcategories)
        )
        subcategory_repr = subcategory_news_repr + subcategory_embedding

        subcategory_clicks = subcategory_mask.sum(dim=1)
        subcategory_weights = subcategory_clicks / subcategory_clicks.sum(
            dim=1
        ).unsqueeze(dim=1).expand(batch_size, self.num_subcategories)

        # Topic-level interest representation
        subcategory_clicks_embedding = self.num_clicks_embedding(subcategory_clicks)
        subcategory_attention_scores = self.subcategory_attention_dense(
            torch.cat((subcategory_repr, subcategory_clicks_embedding), dim=2)
        )
        subcategory_attention_scores = subcategory_attention_scores.expand(
            batch_size, self.num_subcategories, self.num_categories
        ).clone()
        subcategory_attention_scores.masked_fill_(
            ~subcategory_to_category.bool(), 1e-30
        )
        subcategory_attention_weights = F.softmax(subcategory_attention_scores, dim=1)
        category_subcategory_repr = torch.bmm(
            subcategory_attention_weights.transpose(1, 2), subcategory_repr
        )
        category_embedding = self.category_embedding(torch.arange(self.num_categories))
        category_repr = category_subcategory_repr + category_embedding

        category_clicks = category_mask.sum(dim=1)
        category_weights = category_clicks / category_clicks.sum(dim=1).unsqueeze(
            dim=1
        ).expand(batch_size, self.num_categories)

        # User-level interest representation
        category_clicks_embedding = self.num_clicks_embedding(category_clicks)
        category_attention_scores = self.category_attention_dense(
            torch.cat((category_repr, category_clicks_embedding), dim=2)
        )
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
