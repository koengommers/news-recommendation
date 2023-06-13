import torch

from src.models.click_predictors.multi_interest import MultiInterestClickPredictor
from src.models.loss_modules.interest_disagreement import InterestDisagreementLoss
from src.models.loss_modules.prediction import PredictionLoss
from src.models.news_encoders.plm import PLMNewsEncoder
from src.models.news_recommender import NewsRecommender
from src.models.user_encoders.multi_interest import MultiInterestUserEncoder


def init_model(n_interest_vectors):
    news_encoder = PLMNewsEncoder("bert-base-uncased")
    return NewsRecommender(
        news_encoder,
        MultiInterestUserEncoder(
            news_encoder.embedding_dim, n_interest_vectors=n_interest_vectors
        ),
        MultiInterestClickPredictor(news_encoder.embedding_dim),
        [PredictionLoss(), InterestDisagreementLoss()],
    )


def test_news_encoding():
    BATCH_SIZE = 4
    TITLE_LENGTH = 20
    N_INTEREST_VECTORS = 32

    model = init_model(N_INTEREST_VECTORS)

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
    N_INTEREST_VECTORS = 32

    model = init_model(N_INTEREST_VECTORS)

    clicked_news_vector = torch.rand(
        (BATCH_SIZE, N_CLICKED_NEWS, model.news_encoder.config.hidden_size)
    )

    user_vectors = model.encode_user(clicked_news_vector)
    assert isinstance(user_vectors, torch.Tensor)
    assert user_vectors.shape == (
        BATCH_SIZE,
        N_INTEREST_VECTORS,
        model.news_encoder.config.hidden_size,
    )


def test_predicting():
    BATCH_SIZE = 4
    N_CANDIDATE_NEWS = 5
    N_INTEREST_VECTORS = 32

    model = init_model(N_INTEREST_VECTORS)

    news_vector = torch.rand(
        (BATCH_SIZE, N_CANDIDATE_NEWS, model.news_encoder.config.hidden_size)
    )
    user_vectors = torch.rand(
        (BATCH_SIZE, N_INTEREST_VECTORS, model.news_encoder.config.hidden_size)
    )

    prediction = model.rank(news_vector, user_vectors)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (BATCH_SIZE, N_CANDIDATE_NEWS)


def test_forward_pass():
    BATCH_SIZE = 4
    TITLE_LENGTH = 20
    N_CANDIDATE_NEWS = 5
    N_CLICKED_NEWS = 50
    N_INTEREST_VECTORS = 32

    model = init_model(N_INTEREST_VECTORS)

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
