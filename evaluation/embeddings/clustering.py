from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, silhouette_score, v_measure_score


def get_main_category(categories, subcategory):
    return categories.loc[categories["subcategory"] == subcategory].category.values[0]


def evaluate_with_clustering(embeddings, dataset):
    categories = dataset.all_topics

    true_labels = [
        get_main_category(categories, subcategory)
        for subcategory in dataset.topic_encoder.categories_[0]
    ]

    n_clusters = len(set(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
    predicted_labels = kmeans.fit_predict(embeddings)

    return {
        "Rand index": adjusted_rand_score(true_labels, predicted_labels),
        "Mutual information": adjusted_mutual_info_score(true_labels, predicted_labels),
        "Fowlkes-Mallows": fowlkes_mallows_score(true_labels, predicted_labels),
        "V-measure": v_measure_score(true_labels, predicted_labels),
        "Silhouette": silhouette_score(embeddings, predicted_labels),
    }
