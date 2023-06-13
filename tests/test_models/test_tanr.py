import torch

from src.models.click_predictors.inner_product import InnerProductClickPredictor
from src.models.loss_modules.prediction import PredictionLoss
from src.models.loss_modules.topic_prediction import TopicPredictionLoss
from src.models.news_encoders.tanr import TANRNewsEncoder
from src.models.news_recommender import NewsRecommender
from src.models.user_encoders.additive_attention import AdditiveAttentionUserEncoder


def init_model(num_words, num_categories, word_embedding_dim, news_embedding_dim):
    return NewsRecommender(
        TANRNewsEncoder(num_words, word_embedding_dim, num_filters=news_embedding_dim),
        AdditiveAttentionUserEncoder(news_embedding_dim),
        InnerProductClickPredictor(),
        [
            PredictionLoss(),
            TopicPredictionLoss(
                news_embedding_dim=news_embedding_dim, num_categories=num_categories
            ),
        ],
    )


def test_news_encoding():
    NUM_WORDS = 1000
    NUM_CATEGORIES = 20
    WORD_EMBEDDING_DIM = 300
    NEWS_EMBEDDING_DIM = 400
    BATCH_SIZE = 16
    TITLE_LENGTH = 20

    model = init_model(
        NUM_WORDS, NUM_CATEGORIES, WORD_EMBEDDING_DIM, NEWS_EMBEDDING_DIM
    )

    news_article = {
        "title": torch.randint(0, NUM_WORDS, (BATCH_SIZE, TITLE_LENGTH)),
        "category": torch.randint(0, NUM_CATEGORIES, (BATCH_SIZE,)),
    }

    news_vector = model.encode_news(news_article)
    assert isinstance(news_vector, torch.Tensor)
    assert news_vector.shape == (BATCH_SIZE, NEWS_EMBEDDING_DIM)


def test_user_encoding():
    NUM_WORDS = 1000
    NUM_CATEGORIES = 20
    WORD_EMBEDDING_DIM = 300
    NEWS_EMBEDDING_DIM = 400
    BATCH_SIZE = 16
    N_CLICKED_NEWS = 50

    model = init_model(
        NUM_WORDS, NUM_CATEGORIES, WORD_EMBEDDING_DIM, NEWS_EMBEDDING_DIM
    )

    clicked_news_vector = torch.rand((BATCH_SIZE, N_CLICKED_NEWS, NEWS_EMBEDDING_DIM))

    user_vector = model.encode_user(clicked_news_vector)
    assert isinstance(user_vector, torch.Tensor)
    assert user_vector.shape == (BATCH_SIZE, NEWS_EMBEDDING_DIM)


def test_predicting():
    NUM_WORDS = 1000
    NUM_CATEGORIES = 20
    WORD_EMBEDDING_DIM = 300
    NEWS_EMBEDDING_DIM = 400
    BATCH_SIZE = 16
    N_CANDIDATE_NEWS = 5

    model = init_model(
        NUM_WORDS, NUM_CATEGORIES, WORD_EMBEDDING_DIM, NEWS_EMBEDDING_DIM
    )

    news_vector = torch.rand((BATCH_SIZE, N_CANDIDATE_NEWS, NEWS_EMBEDDING_DIM))
    user_vector = torch.rand((BATCH_SIZE, NEWS_EMBEDDING_DIM))

    prediction = model.rank(news_vector, user_vector)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (BATCH_SIZE, N_CANDIDATE_NEWS)


def test_forward_pass():
    NUM_WORDS = 1000
    NUM_CATEGORIES = 20
    WORD_EMBEDDING_DIM = 300
    NEWS_EMBEDDING_DIM = 400
    BATCH_SIZE = 16
    TITLE_LENGTH = 20
    N_CANDIDATE_NEWS = 5
    N_CLICKED_NEWS = 50

    model = init_model(
        NUM_WORDS, NUM_CATEGORIES, WORD_EMBEDDING_DIM, NEWS_EMBEDDING_DIM
    )

    candidate_news = {
        "title": torch.randint(
            0, NUM_WORDS, (BATCH_SIZE, N_CANDIDATE_NEWS, TITLE_LENGTH)
        ),
        "category": torch.randint(0, NUM_CATEGORIES, (BATCH_SIZE, N_CANDIDATE_NEWS)),
    }
    clicked_news = {
        "title": torch.randint(
            0, NUM_WORDS, (BATCH_SIZE, N_CLICKED_NEWS, TITLE_LENGTH)
        ),
        "category": torch.randint(0, NUM_CATEGORIES, (BATCH_SIZE, N_CLICKED_NEWS)),
    }
    labels = torch.zeros(BATCH_SIZE).long()
    loss = model(candidate_news, clicked_news, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
