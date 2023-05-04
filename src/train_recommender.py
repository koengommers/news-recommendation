import random

import hydra
import pandas as pd
import torch
from hydra.utils import to_absolute_path
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

    # Optionally load from checkpoint
    epochs = 0
    if "checkpoint_file" in cfg:
        print(f"Restoring from checkpoint {cfg.checkpoint_file}")
        checkpoint = torch.load(to_absolute_path(cfg.checkpoint_file))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["cpu_rng_state"])
        torch.cuda.set_rng_state(checkpoint["gpu_rng_state"])
        epochs = checkpoint["epochs"]

    dev_metrics_per_epoch = []
    test_metrics_per_epoch = []

    for epoch_num in tqdm(range(epochs + 1, cfg.epochs + 1), disable=cfg.tqdm_disable):
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

        # Evaluate on dev
        dev_metrics, dev_probs = evaluate(
            model, "dev", tokenizer, dataset.categorical_encoders, cfg
        )
        dev_metrics["epoch"] = epoch_num
        dev_metrics_per_epoch.append(dev_metrics)
        dev_probs.to_feather(f"probs_{epoch_num}_dev.feather")

        tqdm.write(
            "(dev) "
            + " | ".join(
                f"{metric}: {dev_metrics[metric]:.5f}" for metric in dev_metrics
            )
        )

        # Evaluate on test
        test_metrics, test_probs = evaluate(
            model, "test", tokenizer, dataset.categorical_encoders, cfg
        )
        test_metrics["epoch"] = epoch_num
        test_metrics_per_epoch.append(test_metrics)
        test_probs.to_feather(f"probs_{epoch_num}_test.feather")

        tqdm.write(
            "(test) "
            + " | ".join(
                f"{metric}: {test_metrics[metric]:.5f}" for metric in test_metrics
            )
        )

        # Save model
        torch.save(
            {
                "epochs": epoch_num,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cpu_rng_state": torch.get_rng_state(),
                "gpu_rng_state": torch.cuda.get_rng_state(),
            },
            f"checkpoint_{epoch_num}.pt",
        )

    # Save metrics
    pd.DataFrame(dev_metrics_per_epoch).to_csv("metrics_dev.csv", index=False)
    pd.DataFrame(test_metrics_per_epoch).to_csv("metrics_test.csv", index=False)


if __name__ == "__main__":
    main()
