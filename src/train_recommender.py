from collections import defaultdict
from enum import Enum

import numpy as np
import torch
import typer
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


def main(
    architecture: Architecture,
    epochs: int = 5,
    mind_variant: str = "small",
    batch_size: int = 64,
    negative_sampling_ratio: int = 4,
    num_words_title: int = 20,
    num_words_abstract: int = 50,
    history_length: int = 50,
    learning_rate: float = 0.0001,
    pretrained_model_name: str = "bert-base-uncased",
    num_batches_show_loss: int = 100,
    use_pretrained_embeddings: bool = True,
    freeze_pretrained_embeddings: bool = False,
    tqdm_disable: bool = False,
):
    # Set up tokenizer
    if architecture in [Architecture.BERT_NRMS, Architecture.MINER]:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
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
    news_features = required_news_features[architecture]
    dataset = RecommenderTrainingDataset(
        mind_variant,
        "train",
        tokenize,
        negative_sampling_ratio,
        num_words_title,
        num_words_abstract,
        history_length,
        news_features,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True
    )

    # Init model
    if architecture == Architecture.NRMS:
        pretrained_embeddings = (
            load_pretrained_embeddings(tokenizer.t2i)
            if use_pretrained_embeddings
            else None
        )
        model = NRMS(
            tokenizer.vocab_size + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
        ).to(device)
    elif architecture == Architecture.TANR:
        pretrained_embeddings = (
            load_pretrained_embeddings(tokenizer.t2i)
            if use_pretrained_embeddings
            else None
        )
        model = TANR(
            tokenizer.vocab_size + 1,
            dataset.categorical_encoders["category"].n_categories + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
        ).to(device)
    elif architecture == Architecture.BERT_NRMS:
        model = BERT_NRMS(pretrained_model_name).to(device)
    elif architecture == Architecture.MINER:
        model = MINER(pretrained_model_name).to(device)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch_num in tqdm(range(1, epochs + 1), disable=tqdm_disable):
        total_train_loss = 0.0

        # Train
        model.train()

        for batch_num, (history, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader), disable=tqdm_disable
        ):
            optimizer.zero_grad()

            labels = torch.zeros(batch_size).long().to(device)
            loss = model(candidate_news, history, labels)
            total_train_loss += loss.item()
            loss.backward()

            optimizer.step()

            if batch_num % num_batches_show_loss == 0:
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
            mind_variant,
            "dev",
            tokenize,
            num_words_title,
            num_words_abstract,
            dataset.categorical_encoders,
            news_features,
        )
        news_dataloader = DataLoader(
            news_dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=False
        )
        news_vectors = {}

        with torch.no_grad():
            for news_ids, batched_news_features in tqdm(
                news_dataloader, desc="Encoding news for evaluation",
                disable=tqdm_disable
            ):
                output = model.get_news_vector(batched_news_features)
                output = output.to(torch.device("cpu"))
                news_vectors.update(dict(zip(news_ids, output)))

        behaviors_dataset = BehaviorsDataset(
            mind_variant,
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
                behaviors_dataset, desc="Evaluating logs", disable=tqdm_disable
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
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
