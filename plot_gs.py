import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from utils import convert_behaviors_to_users, load_behaviors, load_news


def compute_gs_score(embeddings_of_history):
    center = np.mean(embeddings_of_history, axis=0)
    cosine_similarities = [
        np.dot(center, e) / (np.linalg.norm(e) * np.linalg.norm(center))
        for e in embeddings_of_history
    ]
    return np.mean(cosine_similarities)


def get_embeddings_by_ids(ids, news):
    return np.array(news["embedding"][ids].values)


def compute_gs_scores(users, news):
    users["gs"] = (
        users["history"]
        .apply(lambda history: get_embeddings_by_ids(history, news))
        .apply(compute_gs_score)
    )
    return users


def add_news_embeddings(news, embeddings_file):
    with open(embeddings_file, "rb") as f:
        embeddings, topic_encoder = pickle.load(f)

    def encode(x):
        try:
            return topic_encoder.transform([[x]])[0][0]
        except:
            return None

    def embed(x):
        return embeddings[int(x)]

    news["encoded"] = news.subcategory.apply(encode)
    news = news.dropna()
    news["embedding"] = news.encoded.apply(embed)
    return news


def plot_gs(users, dest_file=None):
    users = users[users["history_length"] > 5]
    users["percentile"] = pd.qcut(
        users.history_length, np.linspace(0, 1, 11), labels=np.linspace(0.1, 1, 10)
    )
    fig, ax = plt.subplots()
    for label, df in users.groupby("percentile"):
        df.gs.plot(kind="kde", ax=ax, label=label)
    plt.legend()
    if dest_file:
        plt.savefig(dest_file)
    plt.show()


def main(embeddings_file: str, mind_variant: str = "small"):
    print("Loading behaviors...")
    behaviors = load_behaviors(mind_variant)
    print("Processing users...")
    users = convert_behaviors_to_users(behaviors)
    print(users)
    print(users["history"].apply(len).describe())

    print("Loading news...")
    news = load_news(mind_variant, columns=["id", "category", "subcategory"])
    print("Processing news...")
    news = add_news_embeddings(news, embeddings_file)
    print(news)

    print("Computing GS scores...")
    users = compute_gs_scores(users, news)
    users["history_length"] = users["history"].apply(len)
    users = users.reindex(
        columns=["gs", "topic_count", "subtopic_count", "history_length", "history"]
    )
    print(users)
    print(users["gs"].describe())
    plot_gs(users)


if __name__ == "__main__":
    typer.run(main)
