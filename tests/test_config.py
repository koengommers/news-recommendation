import os
import sys

import hydra

src_path = os.path.abspath(os.path.join("./src"))
if src_path not in sys.path:
    sys.path.append(src_path)


def test_nltk_tokenizer_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender", overrides=["+tokenizer=nltk"]
        )
        assert cfg.tokenizer
        hydra.utils.instantiate(cfg.tokenizer)


def test_bert_tokenizer_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=[
                "+tokenizer=bert",
                "+news_encoder.pretrained_model_name=bert-base-uncased",
            ],
        )
        assert cfg.tokenizer
        assert cfg.tokenizer.pretrained_model_name
        hydra.utils.instantiate(cfg.tokenizer)


def test_nrms_model_config(dataset):
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=["+model=nrms", "news_encoder.use_pretrained_embeddings=false"],
        )
        assert cfg.news_encoder
        assert cfg.model
        news_encoder = hydra.utils.instantiate(cfg.news_encoder, dataset)
        hydra.utils.instantiate(cfg.model, news_encoder, dataset)


def test_tanr_model_config(dataset):
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=["+model=tanr", "news_encoder.use_pretrained_embeddings=false"],
        )
        assert cfg.news_encoder
        assert cfg.model
        news_encoder = hydra.utils.instantiate(cfg.news_encoder, dataset)
        hydra.utils.instantiate(cfg.model, news_encoder, dataset)


def test_bert_nrms_model_config(dataset):
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=[
                "+model=bert-nrms",
            ],
        )
        assert cfg.news_encoder
        assert cfg.model
        news_encoder = hydra.utils.instantiate(cfg.news_encoder, dataset)
        hydra.utils.instantiate(cfg.model, news_encoder, dataset)


def test_miner_model_config(dataset):
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=[
                "+model=miner",
            ],
        )
        assert cfg.news_encoder
        assert cfg.model
        news_encoder = hydra.utils.instantiate(cfg.news_encoder, dataset)
        hydra.utils.instantiate(cfg.model, news_encoder, dataset)
