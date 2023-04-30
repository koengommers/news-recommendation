import random

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.recommender_training import RecommenderTrainingDataset
from evaluation.recommender import evaluate
from models.news_recommender import NewsRecommender
from utils.collate import collate_fn
from utils.context import context
from utils.tokenize import NltkTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    print("========== Config ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================")

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set up tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # Load dataset
    dataset = RecommenderTrainingDataset(
        cfg.mind_variant,
        tokenizer,
        cfg.negative_sampling_ratio,
        cfg.num_words_title,
        cfg.num_words_abstract,
        cfg.history_length,
        cfg.features,
    )
    context.add("num_categories", dataset.num_categories)
    context.add("num_words", dataset.num_words)
    if isinstance(tokenizer, NltkTokenizer):
        context.add("token2int", tokenizer.t2i)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # Init news encoder
    news_encoder = hydra.utils.instantiate(cfg.model.news_encoder)
    context.add("news_embedding_dim", news_encoder.embedding_dim)

    # Init user encoder
    user_encoder = hydra.utils.instantiate(cfg.model.user_encoder)

    # Init click predictor
    click_predictor = hydra.utils.instantiate(cfg.model.click_predictor)

    # Init loss modules
    loss_modules = [
        hydra.utils.instantiate(loss_cfg) for loss_cfg in cfg.model.loss.values()
    ]

    # Init model
    model = NewsRecommender(
        news_encoder, user_encoder, click_predictor, loss_modules
    ).to(device)

    # Init optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    for epoch_num in tqdm(range(1, cfg.epochs + 1), disable=cfg.tqdm_disable):
        total_train_loss = 0.0

        # Train
        model.train()

        for batch_num, (history, mask, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader), disable=cfg.tqdm_disable
        ):
            optimizer.zero_grad()

            labels = torch.zeros(cfg.batch_size).long().to(device)
            if cfg.use_history_mask:
                loss = model(candidate_news, history, labels, mask)
            else:
                loss = model(candidate_news, history, labels)
            total_train_loss += loss.item()
            loss.backward()

            optimizer.step()

            if batch_num % cfg.num_batches_show_loss == 0:
                tqdm.write(
                    f"Loss after {batch_num} batches in epoch {epoch_num}: {total_train_loss / (batch_num)}"
                )

        tqdm.write(
            f"Epochs: {epoch_num} | Average train loss: {total_train_loss / len(dataloader)}"
        )

        # Evaluate
        metrics = evaluate(model, "dev", tokenizer, dataset.categorical_encoders, cfg)

        tqdm.write(
            " | ".join(f"{metric}: {score:.5f}" for metric, score in metrics.items())
        )

        # Save model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            },
            f"checkpoint_{cfg.model.architecture}_{epoch_num}.pt",
        )


if __name__ == "__main__":
    main()
