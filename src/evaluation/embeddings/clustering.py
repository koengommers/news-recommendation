import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import fowlkes_mallows_score, silhouette_score


def get_main_category(categories, subcategory):
    return categories.loc[categories["subcategory"] == subcategory].category.values[0]


def evaluate_with_clustering(embeddings, dataset, n_times=100):
    categories = dataset.all_topics

    true_labels = [
        get_main_category(categories, subcategory)
        for subcategory in dataset.topic_encoder.categories_[0]
    ]
    n_clusters = len(set(true_labels))

    fowlkes_mallows_scores = []
    silhouette_scores = []

    for _ in range(n_times):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        predicted_labels = kmeans.fit_predict(embeddings)
        fowlkes_mallows_scores.append(
            fowlkes_mallows_score(true_labels, predicted_labels)
        )
        silhouette_scores.append(silhouette_score(embeddings, predicted_labels))

    return {
        "FMI": np.mean(fowlkes_mallows_scores),
        "Silhouette": np.mean(silhouette_scores),
    }