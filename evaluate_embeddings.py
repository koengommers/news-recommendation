import numpy as np
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics.pairwise import cosine_similarity


def mrr(y_true, y_scores):
    sorted = np.flip(np.argsort(y_scores), axis=-1)
    ranks = []
    for i in range(len(y_true)):
        rank = np.argwhere(sorted[i] == y_true[i])[0][0]
        ranks.append(rank)
    ranks = np.array(ranks)
    return np.mean(1 / ranks)


def evaluate_embeddings(topic_encoder, embeddings):
    def encode(topic):
        return int(topic_encoder.transform([[topic]])[0][0])

    def embed(topic):
        encoded = encode(topic)
        return embeddings[encoded]

    comparisons = [
        (
            embed("movies-celebrity") - embed("movienews") + embed("musicnews"),
            "music-celebrity",
        ),
        (
            embed("movies-awards") - embed("movienews") + embed("musicnews"),
            "music-awards",
        ),
        (embed("baseball_mlb") - embed("baseball") + embed("mma"), "mmaufc"),
        (embed("celebrity"), "celebritynews"),
        (embed("health-news"), "healthnews"),
    ]

    y_true = []
    y_scores = []

    for vector, target in comparisons:
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
