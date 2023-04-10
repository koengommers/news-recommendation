from collections import defaultdict
import hydra
from omegaconf import DictConfig
from enum import Enum

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.behaviors import BehaviorsDataset
from datasets.news import NewsDataset
from datasets.recommender_training import RecommenderTrainingDataset
from evaluation.metrics import mrr_score, ndcg_score
from models.BERT_NRMS import BERT_NRMS
from models.MINER import MINER
from models.NRMS import NRMS
from models.TANR import TANR
from utils.collate import collate_fn
from utils.data import load_pretrained_embeddings
from utils.tokenize import NltkTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Architecture(str, Enum):
    NRMS = "NRMS"
    BERT_NRMS = "BERT-NRMS"
    TANR = "TANR"
    MINER = "MINER"


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig):
    # Set up tokenizer
    if cfg.model.architecture in [Architecture.BERT_NRMS, Architecture.MINER]:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model_name)
        tokenize = lambda text, length: dict(
            tokenizer(
                text,
                max_length=length,
                padding="max_length",
                truncation=True,
            )
        )
    else:
        tokenizer = NltkTokenizer()
        tokenize = lambda text, length: tokenizer(text, length)

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
        tokenize,
        cfg.negative_sampling_ratio,
        cfg.num_words_title,
        cfg.num_words_abstract,
        cfg.history_length,
        news_features,
    )
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, drop_last=True
    )

    # Init model
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
        model = BERT_NRMS(cfg.model.pretrained_model_name).to(device)
    elif cfg.model.architecture == Architecture.MINER:
        model = MINER(cfg.model.pretrained_model_name).to(device)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    for epoch_num in tqdm(range(1, cfg.epochs + 1), disable=cfg.tqdm_disable):
        total_train_loss = 0.0

        # Train
        model.train()

        for batch_num, (history, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader), disable=cfg.tqdm_disable
        ):
            optimizer.zero_grad()

            labels = torch.zeros(cfg.batch_size).long().to(device)
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
        model.eval()
        if isinstance(tokenizer, NltkTokenizer):
            tokenizer.eval()

        news_dataset = NewsDataset(
            cfg.mind_variant,
            "dev",
            tokenize,
            cfg.num_words_title,
            cfg.num_words_abstract,
            dataset.categorical_encoders,
            news_features,
        )
        news_dataloader = DataLoader(
            news_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, drop_last=False
        )
        news_vectors = {}

        with torch.no_grad():
            for news_ids, batched_news_features in tqdm(
                news_dataloader, desc="Encoding news for evaluation",
                disable=cfg.tqdm_disable
            ):
                output = model.get_news_vector(batched_news_features)
                output = output.to(torch.device("cpu"))
                news_vectors.update(dict(zip(news_ids, output)))

        behaviors_dataset = BehaviorsDataset(
            cfg.mind_variant,
            "dev",
        )

        scoring_functions = {
            "AUC": roc_auc_score,
            "MRR": mrr_score,
            "NDCG@5": lambda y_true, y_score: ndcg_score(y_true, y_score, 5),
            "NDCG@10": lambda y_true, y_score: ndcg_score(y_true, y_score, 10),
        }
        all_scores = defaultdict(list)

        with torch.no_grad():
            for history_ids, impression_ids, clicked in tqdm(
                behaviors_dataset, desc="Evaluating logs", disable=cfg.tqdm_disable
            ):
                if len(history_ids) == 0:
                    continue
                history = torch.stack([news_vectors[id] for id in history_ids])
                impressions = torch.stack([news_vectors[id] for id in impression_ids])
                user_vector = model.get_user_vector(history.unsqueeze(dim=0)).squeeze(
                    dim=0
                )
                probs = model.get_prediction(impressions.to(device), user_vector)
                probs = probs.tolist()
                for metric, scoring_fn in scoring_functions.items():
                    all_scores[metric].append(scoring_fn(clicked, probs))

        tqdm.write(
            " | ".join(
                f"{metric}: {np.mean(scores):.5f}" for metric, scores in all_scores.items()
            )
        )


if __name__ == "__main__":
    main()
