import hydra

from src.models.news_recommender import NewsRecommender


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
                "+tokenizer=plm",
                "+model.news_encoder.pretrained_model_name=bert-base-uncased",
            ],
        )
        assert cfg.tokenizer
        assert cfg.tokenizer.pretrained_model_name
        hydra.utils.instantiate(cfg.tokenizer)


def test_nrms_model_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=["+model=nrms", "model.news_encoder.use_pretrained_embeddings=false"],
        )
        assert cfg.model.news_encoder
        assert cfg.model.user_encoder
        assert cfg.model.click_predictor
        assert cfg.model.loss

        news_encoder = hydra.utils.instantiate(cfg.model.news_encoder, num_words=1000)
        user_encoder = hydra.utils.instantiate(cfg.model.user_encoder)
        click_predictor = hydra.utils.instantiate(cfg.model.click_predictor)
        loss_modules = [
            hydra.utils.instantiate(loss_cfg) for loss_cfg in cfg.model.loss.values()
        ]
        model = NewsRecommender(
            news_encoder, user_encoder, click_predictor, loss_modules
        )

        assert isinstance(model, NewsRecommender)


def test_tanr_model_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=["+model=tanr", "model.news_encoder.use_pretrained_embeddings=false"],
        )
        assert cfg.model.news_encoder
        assert cfg.model.user_encoder
        assert cfg.model.click_predictor
        assert cfg.model.loss

        news_encoder = hydra.utils.instantiate(cfg.model.news_encoder, num_words=1000)
        user_encoder = hydra.utils.instantiate(cfg.model.user_encoder)
        click_predictor = hydra.utils.instantiate(cfg.model.click_predictor)
        loss_modules = [
            hydra.utils.instantiate(loss_cfg) for loss_cfg in cfg.model.loss.values()
        ]
        model = NewsRecommender(
            news_encoder, user_encoder, click_predictor, loss_modules
        )

        assert isinstance(model, NewsRecommender)


def test_bert_nrms_model_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=[
                "+model=bert-nrms",
            ],
        )
        assert cfg.model.news_encoder
        assert cfg.model.user_encoder
        assert cfg.model.click_predictor
        assert cfg.model.loss

        news_encoder = hydra.utils.instantiate(cfg.model.news_encoder)
        user_encoder = hydra.utils.instantiate(cfg.model.user_encoder, news_embedding_dim=news_encoder.embedding_dim)
        click_predictor = hydra.utils.instantiate(cfg.model.click_predictor)
        loss_modules = [
            hydra.utils.instantiate(loss_cfg) for loss_cfg in cfg.model.loss.values()
        ]
        model = NewsRecommender(
            news_encoder, user_encoder, click_predictor, loss_modules
        )

        assert isinstance(model, NewsRecommender)


def test_miner_model_config():
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(
            config_name="train_recommender",
            overrides=[
                "+model=miner",
            ],
        )
        assert cfg.model.news_encoder
        assert cfg.model.user_encoder
        assert cfg.model.click_predictor
        assert cfg.model.loss

        news_encoder = hydra.utils.instantiate(cfg.model.news_encoder)
        user_encoder = hydra.utils.instantiate(cfg.model.user_encoder)
        click_predictor = hydra.utils.instantiate(cfg.model.click_predictor)
        loss_modules = [
            hydra.utils.instantiate(loss_cfg) for loss_cfg in cfg.model.loss.values()
        ]
        model = NewsRecommender(
            news_encoder, user_encoder, click_predictor, loss_modules
        )

        assert isinstance(model, NewsRecommender)
