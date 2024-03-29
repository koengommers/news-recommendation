import random
from collections import defaultdict

import hydra
import pandas as pd
import pyrootutils
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# make linter ignore "Module level import not at top of file"
# ruff: noqa: E402
from src.datasets.recommender_training import RecommenderTrainingDataset
from src.evaluation.recommender import evaluate
from src.models.news_recommender import NewsRecommender
from src.utils.collate import collate_fn
from src.utils.context import context
from src.utils.hydra import print_config
from src.utils.tokenize import NltkTokenizer


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    print_config(cfg)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_type)

    # Set up tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # Load dataset
    dataset = RecommenderTrainingDataset(
        cfg.data.mind_variant,
        tokenizer,
        cfg.negative_sampling_ratio,
        cfg.num_words_title,
        cfg.num_words_abstract,
        cfg.history_length,
        cfg.features,
    )
    context.add("num_categories", dataset.num_categories)
    context.add("num_subcategories", dataset.num_subcategories)
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

    amp_enabled = cfg.enable_amp and device_type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)

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

    scheduler = (
        hydra.utils.instantiate(
            cfg.lr_scheduler,
            optimizer,
            epochs=cfg.epochs,
            last_epoch=epochs - 1,
            steps_per_epoch=len(dataloader),
        )
        if "lr_scheduler" in cfg
        else None
    )

    metrics_per_epoch = defaultdict(list)

    for epoch_num in tqdm(range(epochs + 1, cfg.epochs + 1)):
        total_train_loss = 0.0

        # Train
        model.train()

        for batch_num, (history, mask, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader)
        ):
            optimizer.zero_grad()

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                labels = torch.zeros(cfg.batch_size).long().to(device)
                if cfg.use_history_mask:
                    loss = model(candidate_news, history, labels, mask)
                else:
                    loss = model(candidate_news, history, labels)
                total_train_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            if cfg.num_batches_show_loss and batch_num % cfg.num_batches_show_loss == 0:
                tqdm.write(
                    f"Average loss in epoch {epoch_num} after {batch_num} batches: {total_train_loss / (batch_num)}"
                )

        tqdm.write(
            f"Epochs: {epoch_num} | Average train loss: {total_train_loss / len(dataloader)}"
        )

        # Evaluate
        for split in cfg.eval_splits:
            metrics, probs = evaluate(
                model, split, tokenizer, dataset.categorical_encoders, cfg
            )
            metrics["epoch"] = epoch_num
            metrics_per_epoch[split].append(metrics)
            probs.to_feather(f"probs_{epoch_num}_{split}.feather")

            tqdm.write(
                f"({split}) "
                + " | ".join(f"{metric}: {metrics[metric]:.5f}" for metric in metrics)
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
    for split in metrics_per_epoch:
        pd.DataFrame(metrics_per_epoch[split]).to_csv(
            f"metrics_{split}.csv", index=False
        )


if __name__ == "__main__":
    main()
