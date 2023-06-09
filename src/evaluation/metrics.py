from typing import Optional

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def mrr(y_true: list[int], y_scores: list[float]) -> np.floating:
    sorted = np.flip(np.argsort(y_scores), axis=-1)
    ranks: list[int] = []
    for i in range(len(y_true)):
        rank = np.argwhere(sorted[i] == y_true[i])[0][0] + 1
        ranks.append(rank)
    ranks_np = np.array(ranks)
    return np.mean(1 / ranks_np)


def gs_score(embeddings: np.ndarray, distance_metric="cosine") -> Optional[np.floating]:
    if len(embeddings) == 0:
        return None

    center = np.mean(embeddings, axis=0)
    cosine_similarities = pairwise_distances(
        embeddings, center.reshape(1, -1), metric=distance_metric
    )
    return cosine_similarities.mean()


def ild(embeddings, distance_metric="cosine"):
    if len(embeddings) == 0:
        return None
    if len(embeddings) == 1:
        return 0
    distances = pairwise_distances(embeddings, metric=distance_metric)
    triu_indices = np.triu_indices_from(distances, k=1)
    return distances[triu_indices].mean()


# taken from original MIND evaluation script: https://github.com/msnews/MIND/blob/master/evaluate.py
# only added type annotations
def dcg_score(y_true: list[int], y_score: list[int], k: int = 10) -> np.floating:
    order = np.argsort(y_score)[::-1]
    result = np.take(y_true, order[:k])
    gains = 2**result - 1
    discounts = np.log2(np.arange(len(result)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true: list[int], y_score: list[int], k: int = 10) -> np.floating:
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best


def mrr_score(y_true: list[int], y_score: list[int]) -> np.floating:
    order = np.argsort(y_score)[::-1]
    result = np.take(y_true, order)
    rr_score = result / (np.arange(len(result)) + 1)
    return np.sum(rr_score) / np.sum(y_true)
