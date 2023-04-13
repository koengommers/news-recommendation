from itertools import product

import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from evaluation.metrics import mrr

TOPIC_GROUPS = [
    # Media and their celebrities
    [
        ("movienews", "movies-celebrity"),
        ("musicnews", "music-celebrity"),
        ("tvnews", "tv-celebrity"),
    ],
    # Sports and their leagues
    [
        ("mma", "mmaufc"),
        ("baseball", "baseball_mlb"),
    ],
]


def evaluate_with_analogies(topic_encoder: LabelEncoder, embeddings: np.ndarray) -> dict[str, np.floating]:
    def encode(topic: str) -> int:
        return int(topic_encoder.transform([topic])[0])

    def embed(topic: str) -> np.ndarray:
        encoded = encode(topic)
        return embeddings[encoded]

    analogies = []
    for group in TOPIC_GROUPS:
        for a, b in product(group, repeat=2):
            if a != b:
                analogies.append((embed(a[1]) - embed(a[0]) + embed(b[0]), b[1]))

    y_true = []
    y_scores = []

    for vector, target in analogies:
        cosine_similarities = cosine_similarity(vector.reshape(1, -1), embeddings)[0]
        target_index = encode(target)
        y_true.append(target_index)
        y_scores.append(cosine_similarities)

    labels = list(range(0, len(embeddings)))
    return {
        "P@1": top_k_accuracy_score(y_true, y_scores, k=1, labels=labels),
        "P@5": top_k_accuracy_score(y_true, y_scores, k=5, labels=labels),
        "MRR": mrr(y_true, y_scores),
    }
