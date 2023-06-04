import pytest
import torch
from src.models.click_predictors.hierec import HieRecClickPredictor

from src.models.user_encoders.hierec import HieRecUserEncoder


def test_user_encoder():
    batch_size = 4
    history_length = 40
    news_embedding_dim = 300
    categories_embed_dim = 300
    num_clicked_embed_dim = 50
    num_categories = 15
    num_subcategories = 200

    news = {
        "vectors": torch.rand((batch_size, history_length, news_embedding_dim)),
        "category": torch.randint(0, num_categories, (batch_size, history_length)),
        "subcategory": torch.randint(
            0, num_subcategories, (batch_size, history_length)
        ),
    }

    user_encoder = HieRecUserEncoder(
        num_subcategories,
        num_categories,
        news_embedding_dim,
        categories_embed_dim,
        history_length,
        num_clicked_embed_dim,
    )

    (
        subcategory_repr,
        subcategory_weights,
        category_repr,
        category_weights,
        user_repr,
    ) = user_encoder(news)

    assert subcategory_repr.shape == (
        batch_size,
        num_subcategories,
        categories_embed_dim,
    )
    assert subcategory_weights.shape == (
        batch_size,
        num_subcategories,
    )
    assert category_repr.shape == (batch_size, num_categories, categories_embed_dim)
    assert category_weights.shape == (
        batch_size,
        num_categories,
    )
    assert user_repr.shape == (batch_size, categories_embed_dim)

def test_click_predictor():
    batch_size = 4
    n_candidate_news = 5
    news_embedding_dim = 300
    categories_embed_dim = 300
    num_categories = 15
    num_subcategories = 200
    lambda_t = 0.15
    lambda_s = 0.7

    candidate_news = {
        "vectors": torch.rand((batch_size, n_candidate_news, news_embedding_dim)),
        "category": torch.randint(0, num_categories, (batch_size, n_candidate_news)),
        "subcategory": torch.randint(
            0, num_subcategories, (batch_size, n_candidate_news)
        ),
    }

    user_repr = (
        torch.rand((batch_size, num_subcategories, categories_embed_dim)),
        torch.rand((batch_size, num_subcategories)),
        torch.rand((batch_size, num_categories, categories_embed_dim)),
        torch.rand((batch_size, num_categories)),
        torch.rand((batch_size, categories_embed_dim))
    )

    click_predictor = HieRecClickPredictor(lambda_t, lambda_s)

    probs = click_predictor(candidate_news, user_repr)

    assert probs.shape == (batch_size, n_candidate_news)
