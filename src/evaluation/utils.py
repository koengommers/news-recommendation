import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def find_closest_embedding(
    embeddings: np.ndarray, topic_encoder: LabelEncoder, topic: str, n: int = 5
) -> list[str]:
    def encode(topic: str) -> int:
        return int(topic_encoder.transform([topic])[0])

    def embed(topic: str) -> np.ndarray:
        encoded = encode(topic)
        return embeddings[encoded]

    target = embed(topic)
    cosine_similarities = cosine_similarity(target.reshape(1, -1), embeddings)[0]
    sorted = np.flip(np.argsort(cosine_similarities))
    closest_topics = topic_encoder.inverse_transform(sorted[1 : 1 + n])
    assert closest_topics is not None
    return list(closest_topics)


def print_closest_topics(
    embeddings: np.ndarray, topic_encoder: LabelEncoder, topics: list[str], n: int = 5
) -> None:
    tqdm.write("== Closest topics ==")
    for topic in topics:
        closest_topics = find_closest_embedding(embeddings, topic_encoder, topic, n)
        tqdm.write(f"{topic}: {', '.join(closest_topics)}")
