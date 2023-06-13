import torch

from src.models.click_predictors.inner_product import InnerProductClickPredictor
from src.models.loss_modules.prediction import PredictionLoss
from src.models.news_encoders.plm import PLMNewsEncoder
from src.models.news_recommender import NewsRecommender
from src.models.user_encoders.nrms import NRMSUserEncoder


def init_model():
    news_encoder = PLMNewsEncoder("bert-base-uncased")
    return NewsRecommender(
        news_encoder,
        NRMSUserEncoder(news_encoder.embedding_dim, num_attention_heads=16),
        InnerProductClickPredictor(),
        PredictionLoss(),
    )


def test_news_encoding():
    BATCH_SIZE = 4
    TITLE_LENGTH = 20

    model = init_model()

    news_article = {
        "title": {
            "input_ids": torch.randint(
                0, model.news_encoder.config.vocab_size, (BATCH_SIZE, TITLE_LENGTH)
            ),
            "attention_mask": torch.randint(0, 1, (BATCH_SIZE, TITLE_LENGTH)),
            "token_type_ids": torch.zeros(
                (BATCH_SIZE, TITLE_LENGTH), dtype=torch.int64
            ),
        }
    }

    news_vector = model.encode_news(news_article)
    assert isinstance(news_vector, torch.Tensor)
    assert news_vector.shape == (BATCH_SIZE, model.news_encoder.config.hidden_size)


def test_user_encoding():
    BATCH_SIZE = 4
    N_CLICKED_NEWS = 50

    model = init_model()

    clicked_news_vector = torch.rand(
        (BATCH_SIZE, N_CLICKED_NEWS, model.news_encoder.config.hidden_size)
    )

    user_vector = model.encode_user(clicked_news_vector)
    assert isinstance(user_vector, torch.Tensor)
    assert user_vector.shape == (BATCH_SIZE, model.news_encoder.config.hidden_size)


def test_predicting():
    BATCH_SIZE = 4
    N_CANDIDATE_NEWS = 5

    model = init_model()

    news_vector = torch.rand(
        (BATCH_SIZE, N_CANDIDATE_NEWS, model.news_encoder.config.hidden_size)
    )
    user_vector = torch.rand((BATCH_SIZE, model.news_encoder.config.hidden_size))

    prediction = model.rank(news_vector, user_vector)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (BATCH_SIZE, N_CANDIDATE_NEWS)


def test_forward_pass():
    BATCH_SIZE = 4
    TITLE_LENGTH = 20
    N_CANDIDATE_NEWS = 5
    N_CLICKED_NEWS = 50

    model = init_model()

    candidate_news = {
        "title": {
            "input_ids": torch.randint(
                0,
                model.news_encoder.config.vocab_size,
                (BATCH_SIZE, N_CANDIDATE_NEWS, TITLE_LENGTH),
            ),
            "attention_mask": torch.randint(
                0, 1, (BATCH_SIZE, N_CANDIDATE_NEWS, TITLE_LENGTH)
            ),
            "token_type_ids": torch.zeros(
                (BATCH_SIZE, N_CANDIDATE_NEWS, TITLE_LENGTH), dtype=torch.int64
            ),
        }
    }
    clicked_news = {
        "title": {
            "input_ids": torch.randint(
                0,
                model.news_encoder.config.vocab_size,
                (BATCH_SIZE, N_CLICKED_NEWS, TITLE_LENGTH),
            ),
            "attention_mask": torch.randint(
                0, 1, (BATCH_SIZE, N_CLICKED_NEWS, TITLE_LENGTH)
            ),
            "token_type_ids": torch.zeros(
                (BATCH_SIZE, N_CLICKED_NEWS, TITLE_LENGTH), dtype=torch.int64
            ),
        }
    }
    labels = torch.zeros(BATCH_SIZE).long()
    loss = model(candidate_news, clicked_news, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
