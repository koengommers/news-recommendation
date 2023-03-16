import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             fowlkes_mallows_score, silhouette_score,
                             v_measure_score)


def get_main_category(categories, subcategory):
    return categories.loc[categories["subcategory"] == subcategory].category.values[0]


def evaluate_with_clustering(embeddings, dataset, n_times=100):
    categories = dataset.all_topics

    true_labels = [
        get_main_category(categories, subcategory)
        for subcategory in dataset.topic_encoder.categories_[0]
    ]
    n_clusters = len(set(true_labels))

    rand_scores = []
    mutual_info_scores = []
    fowlkes_mallows_scores = []
    v_measure_scores = []
    silhouette_scores = []

    for _ in range(n_times):
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        predicted_labels = kmeans.fit_predict(embeddings)
        rand_scores.append(adjusted_rand_score(true_labels, predicted_labels))
        mutual_info_scores.append(
            adjusted_mutual_info_score(true_labels, predicted_labels)
        )
        fowlkes_mallows_scores.append(
            fowlkes_mallows_score(true_labels, predicted_labels)
        )
        v_measure_scores.append(v_measure_score(true_labels, predicted_labels))
        silhouette_scores.append(silhouette_score(embeddings, predicted_labels))

    return {
        "Rand index": np.mean(rand_scores),
        "Mutual information": np.mean(mutual_info_scores),
        "Fowlkes-Mallows": np.mean(fowlkes_mallows_scores),
        "V-measure": np.mean(v_measure_scores),
        "Silhouette": np.mean(silhouette_scores),
    }
