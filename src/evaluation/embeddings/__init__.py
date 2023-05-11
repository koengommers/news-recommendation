import numpy as np

from datasets.topic_reads import TopicReadsDataset
from evaluation.embeddings.analogies import evaluate_with_analogies
from evaluation.embeddings.clustering import evaluate_with_clustering


def evaluate_embeddings(embeddings: np.ndarray, dataset: TopicReadsDataset):
    metrics = {}
    metrics.update(evaluate_with_analogies(dataset.topic_encoder, embeddings))
    metrics.update(evaluate_with_clustering(embeddings, dataset))
    return metrics
