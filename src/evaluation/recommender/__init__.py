from multiprocessing import Pool
from typing import Callable, Union

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.behaviors import BehaviorsDataset, behaviors_collate_fn
from datasets.news import NewsDataset
from evaluation.metrics import mrr_score, ndcg_score
from models.BERT_NRMS import BERT_NRMS
from models.MINER import MINER
from models.NRMS import NRMS
from models.TANR import TANR
from utils.collate import collate_fn
from utils.encode import CategoricalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TokenizerOutput = Union[list[int], dict[str, list[int]]]


scoring_functions = {
    "AUC": roc_auc_score,
    "MRR": mrr_score,
    "NDCG@5": lambda y_true, y_score: ndcg_score(y_true, y_score, 5),
    "NDCG@10": lambda y_true, y_score: ndcg_score(y_true, y_score, 10),
}


def calculate_metrics(result):
    return {
        metric: scoring_fn(result[0], result[1])
        for metric, scoring_fn in scoring_functions.items()
    }


def evaluate(
    model: Union[NRMS, TANR, BERT_NRMS, MINER],
    split: str,
    tokenizer: Callable[[str, int], TokenizerOutput],
    categorical_encoders: dict[str, CategoricalEncoder],
    news_features: list[str],
    cfg: DictConfig,
) -> dict[str, np.floating]:
    model.eval()
    tokenizer.eval()

    news_dataset = NewsDataset(
        cfg.mind_variant,
        split,
        tokenizer,
        cfg.num_words_title,
        cfg.num_words_abstract,
        categorical_encoders,
        news_features,
    )
    news_dataloader = DataLoader(
        news_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
    )
    news_vectors = {}

    # Precompute news vectors
    with torch.no_grad():
        for news_ids, batched_news_features in tqdm(
            news_dataloader,
            desc="Encoding news for evaluation",
            disable=cfg.tqdm_disable,
        ):
            output = model.get_news_vector(batched_news_features)
            output = output.to(torch.device("cpu"))
            news_vectors.update(dict(zip(news_ids, output)))

    behaviors_dataset = BehaviorsDataset(cfg.mind_variant, split, news_vectors)
    behaviors_dataloader = DataLoader(
        behaviors_dataset,
        batch_size=cfg.batch_size,
        collate_fn=behaviors_collate_fn,
        drop_last=False,
        pin_memory=True,
        num_workers=cfg.num_workers,
    )

    # Make predictions
    results = []
    with torch.no_grad():
        for clicked_news_vectors, mask, impression_ids, clicked in tqdm(
            behaviors_dataloader, desc="Evaluating logs", disable=cfg.tqdm_disable
        ):
            if cfg.use_history_mask:
                user_vectors = model.get_user_vector(clicked_news_vectors, mask)
            else:
                user_vectors = model.get_user_vector(clicked_news_vectors)

            for i in range(len(impression_ids)):
                impressions = torch.stack(
                    [news_vectors[id] for id in impression_ids[i]]
                ).unsqueeze(0)
                probs = model.get_prediction(
                    impressions.to(device), user_vectors[i].unsqueeze(0)
                ).squeeze(0)
                probs_list = probs.tolist()
                results.append((clicked[i], probs_list))

    # Calculate metrics
    with Pool(processes=cfg.num_workers) as pool:
        scores = pool.map(calculate_metrics, results)

    metrics = {
        metric: np.mean([x[metric] for x in scores]) for metric in scoring_functions
    }
    return metrics
