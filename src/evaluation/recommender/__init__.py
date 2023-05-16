from multiprocessing import Pool
from typing import Union

import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.behaviors import BehaviorsDataset, behaviors_collate_fn
from src.datasets.news import NewsDataset
from src.evaluation.metrics import mrr_score, ndcg_score
from src.models.news_recommender import NewsRecommender
from src.utils.collate import collate_fn
from src.utils.encode import CategoricalEncoder
from src.utils.tokenize import BertTokenizer, NltkTokenizer


TokenizerOutput = Union[list[int], dict[str, list[int]]]


scoring_functions = {
    "AUC": roc_auc_score,
    "MRR": mrr_score,
    "NDCG@5": lambda y_true, y_score: ndcg_score(y_true, y_score, 5),
    "NDCG@10": lambda y_true, y_score: ndcg_score(y_true, y_score, 10),
}


def calculate_metrics(result):
    for metric, scoring_fn in scoring_functions.items():
        result[metric] = scoring_fn(result["clicked"], result["probs"])
    return result


def evaluate(
    model: NewsRecommender,
    split: str,
    tokenizer: Union[NltkTokenizer, BertTokenizer],
    categorical_encoders: dict[str, CategoricalEncoder],
    cfg: DictConfig,
) -> tuple[dict[str, float], pd.DataFrame]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    tokenizer.eval()

    news_dataset = NewsDataset(
        cfg.data.mind_variant,
        split,
        tokenizer,
        cfg.num_words_title,
        cfg.num_words_abstract,
        categorical_encoders,
        cfg.features,
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
        ):
            output = model.encode_news(batched_news_features)
            output = output.to(torch.device("cpu"))
            news_vectors.update(dict(zip(news_ids, output)))

    behaviors_dataset = BehaviorsDataset(cfg.data.mind_variant, split, news_vectors)
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
        for log_ids, clicked_news_vectors, mask, impression_ids, clicked in tqdm(
            behaviors_dataloader, desc="Evaluating logs"
        ):
            if cfg.use_history_mask:
                user_vectors = model.encode_user(clicked_news_vectors, mask)
            else:
                user_vectors = model.encode_user(clicked_news_vectors)

            for i in range(len(log_ids)):
                impressions = torch.stack(
                    [news_vectors[id] for id in impression_ids[i]]
                ).unsqueeze(0)
                probs = model.rank(
                    impressions.to(device), user_vectors[i].unsqueeze(0)
                ).squeeze(0)
                probs_list = probs.tolist()
                results.append(
                    {"log_id": log_ids[i], "clicked": clicked[i], "probs": probs_list}
                )

    # Calculate metrics
    with Pool(processes=cfg.num_workers) as pool:
        scores = pool.map(calculate_metrics, results)

    eval_data = pd.DataFrame(scores)
    eval_data = eval_data.set_index("log_id")

    metrics = {metric: eval_data[metric].mean() for metric in scoring_functions}
    probs = eval_data["probs"].reset_index()

    return metrics, probs
