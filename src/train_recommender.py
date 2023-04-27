import random
from enum import Enum
from typing import Union

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.recommender_training import RecommenderTrainingDataset
from evaluation.recommender import evaluate
from models.BERT_NRMS import BERT_NRMS
from models.MINER import MINER
from models.NRMS import NRMS
from models.TANR import TANR
from utils.collate import collate_fn
from utils.data import load_pretrained_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Architecture(str, Enum):
    NRMS = "NRMS"
    BERT_NRMS = "BERT-NRMS"
    TANR = "TANR"
    MINER = "MINER"


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Set up tokenizer
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    # Load dataset
    required_news_features = {
        Architecture.NRMS: ["title"],
        Architecture.BERT_NRMS: ["title"],
        Architecture.TANR: ["title", "category"],
        Architecture.MINER: ["title"],
    }
    news_features = required_news_features[cfg.model.architecture]
    dataset = RecommenderTrainingDataset(
        cfg.mind_variant,
        "train",
        tokenizer,
        cfg.negative_sampling_ratio,
        cfg.num_words_title,
        cfg.num_words_abstract,
        cfg.history_length,
        news_features,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # Init model
    model: Union[NRMS, TANR, BERT_NRMS, MINER]
    if cfg.model.architecture == Architecture.NRMS:
        pretrained_embeddings = (
            load_pretrained_embeddings(tokenizer.t2i)
            if cfg.model.use_pretrained_embeddings
            else None
        )
        model = NRMS(
            tokenizer.vocab_size + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=cfg.model.freeze_pretrained_embeddings,
        ).to(device)
    elif cfg.model.architecture == Architecture.TANR:
        pretrained_embeddings = (
            load_pretrained_embeddings(tokenizer.t2i)
            if cfg.model.use_pretrained_embeddings
            else None
        )
        model = TANR(
            tokenizer.vocab_size + 1,
            dataset.categorical_encoders["category"].n_categories + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=cfg.model.freeze_pretrained_embeddings,
        ).to(device)
    elif cfg.model.architecture == Architecture.BERT_NRMS:
        model = BERT_NRMS(
            cfg.model.pretrained_model_name,
            num_hidden_layers=cfg.model.num_hidden_layers,
        ).to(device)
    elif cfg.model.architecture == Architecture.MINER:
        model = MINER(cfg.model.pretrained_model_name).to(device)
    else:
        raise ValueError("Unknown model architecture")

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
        if isinstance(tokenizer, NltkTokenizer):
            tokenizer.eval()

        metrics = evaluate(
            model, "dev", tokenize, dataset.categorical_encoders, news_features, cfg
        )

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
