import os
import sys

import torch

src_path = os.path.abspath(os.path.join("./src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from src.models.modules.bert.news_encoder import NewsEncoder
from src.models.NRMS import NRMS


def init_model(dataset):
    news_encoder = NewsEncoder(dataset)
    model = NRMS(news_encoder, dataset, num_attention_heads=16)
    return model


def test_news_encoding(dataset):
    BATCH_SIZE = 4
    TITLE_LENGTH = 20

    model = init_model(dataset)

    news_article = {
        "title": {
            "input_ids": torch.randint(
                0, model.news_encoder.bert_config.vocab_size, (BATCH_SIZE, TITLE_LENGTH)
            ),
            "attention_mask": torch.randint(0, 1, (BATCH_SIZE, TITLE_LENGTH)),
            "token_type_ids": torch.zeros(
                (BATCH_SIZE, TITLE_LENGTH), dtype=torch.int64
            ),
        }
    }

    news_vector = model.get_news_vector(news_article)
    assert isinstance(news_vector, torch.Tensor)
    assert news_vector.shape == (BATCH_SIZE, model.news_encoder.bert_config.hidden_size)


def test_user_encoding(dataset):
    BATCH_SIZE = 4
    N_CLICKED_NEWS = 50

    model = init_model(dataset)

    clicked_news_vector = torch.rand(
        (BATCH_SIZE, N_CLICKED_NEWS, model.news_encoder.bert_config.hidden_size)
    )

    user_vector = model.get_user_vector(clicked_news_vector)
    assert isinstance(user_vector, torch.Tensor)
    assert user_vector.shape == (BATCH_SIZE, model.news_encoder.bert_config.hidden_size)


def test_predicting(dataset):
    BATCH_SIZE = 4
    N_CANDIDATE_NEWS = 5

    model = init_model(dataset)

    news_vector = torch.rand(
        (BATCH_SIZE, N_CANDIDATE_NEWS, model.news_encoder.bert_config.hidden_size)
    )
    user_vector = torch.rand((BATCH_SIZE, model.news_encoder.bert_config.hidden_size))

    prediction = model.get_prediction(news_vector, user_vector)
    assert isinstance(prediction, torch.Tensor)
    assert prediction.shape == (BATCH_SIZE, N_CANDIDATE_NEWS)


def test_forward_pass(dataset):
    BATCH_SIZE = 4
    TITLE_LENGTH = 20
    N_CANDIDATE_NEWS = 5
    N_CLICKED_NEWS = 50

    model = init_model(dataset)

    candidate_news = {
        "title": {
            "input_ids": torch.randint(
                0,
                model.news_encoder.bert_config.vocab_size,
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
                model.news_encoder.bert_config.vocab_size,
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
