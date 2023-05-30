import random

import hydra
import numpy as np
import pyrootutils
import torch
from omegaconf import DictConfig
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# make linter ignore "Module level import not at top of file"
# ruff: noqa: E402
from src.datasets.topic_prediction import TopicPredictionDataset
from src.models.topic_predictor import TopicPredictor
from src.utils.collate import collate_fn
from src.utils.context import context
from src.utils.hydra import print_config
from src.utils.tokenize import NltkTokenizer


@hydra.main(
    version_base=None, config_path="../conf", config_name="train_topic_predictor"
)
def main(cfg: DictConfig) -> None:
    print_config(cfg)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # Load dataset
    if "category" in cfg.features:
        print("Warning: using category to predict category")
    dataset = TopicPredictionDataset(
        cfg.data.mind_variant,
        tokenizer,
        cfg.num_words_title,
        cfg.num_words_abstract,
        news_features=cfg.features,
    )
    context.add("num_categories", dataset.num_categories)
    context.add("num_words", dataset.num_words)
    if isinstance(tokenizer, NltkTokenizer):
        context.add("token2int", tokenizer.t2i)

    train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # Init model
    news_encoder = hydra.utils.instantiate(cfg.model.news_encoder)
    model = TopicPredictor(news_encoder, dataset.num_categories).to(device)

    # Init optimizer
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch_num in tqdm(range(1, cfg.epochs + 1)):
        model.train()
        running_loss = 0.0

        for batch_num, (news, categories) in tqdm(
            enumerate(train_dataloader, 1), total=len(train_dataloader)
        ):
            optimizer.zero_grad()

            probs = model(news)
            loss = loss_fn(probs, categories.to(device))
            running_loss += loss.item()

            loss.backward()
            optimizer.step()

            if cfg.num_batches_show_loss and batch_num % cfg.num_batches_show_loss == 0:
                average_loss = running_loss / cfg.num_batches_show_loss
                tqdm.write(
                    f"Average loss in epoch {epoch_num}"
                    f" for batches {batch_num - cfg.num_batches_show_loss} - {batch_num}:"
                    f" {average_loss}"
                )
                running_loss = 0.0

        model.eval()
        all_probs: list[list[int]] = []
        all_categories: list[int] = []
        with torch.no_grad():
            for news, categories in tqdm(test_dataloader):
                probs = model(news)
                all_probs = all_probs + probs.tolist()
                all_categories = all_categories + categories.tolist()

        metrics = {
            "accuracy": accuracy_score,
            "balanced accuracy": balanced_accuracy_score,
        }
        results = {
            key: fn(all_categories, np.argmax(all_probs, axis=1))
            for key, fn in metrics.items()
        }
        tqdm.write(" | ".join(f"{metric}: {results[metric]:.5f}" for metric in results))


if __name__ == "__main__":
    main()
