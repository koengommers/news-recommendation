from evaluation.embeddings.analogies import evaluate_with_analogies
from evaluation.embeddings.clustering import evaluate_with_clustering


def evaluate_embeddings(embeddings, dataset):
    metrics = {}
    metrics.update(evaluate_with_analogies(dataset.topic_encoder, embeddings))
    metrics.update(evaluate_with_clustering(embeddings, dataset))
    return metrics
