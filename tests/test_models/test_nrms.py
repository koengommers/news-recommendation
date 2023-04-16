import os
import sys

import torch

src_path = os.path.abspath(os.path.join("./src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.models.NRMS import NRMS


def test_news_encoding():
    NUM_WORDS = 1000
    WORD_EMBEDDING_DIM = 300
    BATCH_SIZE = 16
    TITLE_LENGTH = 20

    model = NRMS(NUM_WORDS, WORD_EMBEDDING_DIM)

    news_article = {"title": torch.randint(0, NUM_WORDS, (BATCH_SIZE, TITLE_LENGTH))}

    news_vector = model.get_news_vector(news_article)
    assert isinstance(news_vector, torch.Tensor)
    assert news_vector.shape == (BATCH_SIZE, WORD_EMBEDDING_DIM)


def test_user_encoding():
    NUM_WORDS = 1000
    WORD_EMBEDDING_DIM = 300
    BATCH_SIZE = 16
    N_CLICKED_NEWS = 50

    model = NRMS(NUM_WORDS, WORD_EMBEDDING_DIM)

    clicked_news_vector = torch.rand((BATCH_SIZE, N_CLICKED_NEWS, WORD_EMBEDDING_DIM))

    user_vector = model.get_user_vector(clicked_news_vector)
    assert isinstance(user_vector, torch.Tensor)
    assert user_vector.shape == (BATCH_SIZE, WORD_EMBEDDING_DIM)


def test_predicting():
    NUM_WORDS = 1000
    WORD_EMBEDDING_DIM = 300
    BATCH_SIZE = 16
    N_CANDIDATE_NEWS = 5

    model = NRMS(NUM_WORDS, WORD_EMBEDDING_DIM)

    news_vector = torch.rand((BATCH_SIZE, N_CANDIDATE_NEWS, WORD_EMBEDDING_DIM))
    user_vector = torch.rand((BATCH_SIZE, WORD_EMBEDDING_DIM,))

    prediction = model.get_prediction(news_vector, user_vector)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (BATCH_SIZE, N_CANDIDATE_NEWS)


def test_forward_pass():
    NUM_WORDS = 1000
    WORD_EMBEDDING_DIM = 300
    BATCH_SIZE = 16
    TITLE_LENGTH = 20
    N_CANDIDATE_NEWS = 5
    N_CLICKED_NEWS = 50

    model = NRMS(NUM_WORDS, WORD_EMBEDDING_DIM)

    candidate_news = {
        "title": torch.randint(
            0, NUM_WORDS, (BATCH_SIZE, N_CANDIDATE_NEWS, TITLE_LENGTH)
        )
    }
    clicked_news = {
        "title": torch.randint(0, NUM_WORDS, (BATCH_SIZE, N_CLICKED_NEWS, TITLE_LENGTH))
    }
    labels = torch.zeros(BATCH_SIZE).long()
    loss = model(candidate_news, clicked_news, labels)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
