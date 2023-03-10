from itertools import product

import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def mrr(y_true, y_scores):
    sorted = np.flip(np.argsort(y_scores), axis=-1)
    ranks = []
    for i in range(len(y_true)):
        rank = np.argwhere(sorted[i] == y_true[i])[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return np.mean(1 / ranks)


def evaluate_embeddings(topic_encoder, embeddings):
    def encode(topic):
        return int(topic_encoder.transform([[topic]])[0][0])

    def embed(topic):
        encoded = encode(topic)
        return embeddings[encoded]

    topic_classes = [
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
    analogies = []
    for cls in topic_classes:
        for a, b in product(cls, repeat=2):
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
