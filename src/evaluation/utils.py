import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def find_closest_embedding(embeddings, topic_encoder, topic, n=5):
    def encode(topic):
        return int(topic_encoder.transform([[topic]])[0][0])

    def embed(topic):
        encoded = encode(topic)
        return embeddings[encoded]

    target = embed(topic)
    cosine_similarities = cosine_similarity(target.reshape(1, -1), embeddings)[0]
    sorted = np.flip(np.argsort(cosine_similarities))
    return list(
        topic_encoder.inverse_transform(sorted[1 : 1 + n].reshape(-1, 1)).reshape(-1)
    )

def print_closest_topics(embeddings, topic_encoder, topics, n=5):
    tqdm.write("== Closest topics ==")
    for topic in topics:
        closest_topics = find_closest_embedding(
            embeddings, topic_encoder, topic, n
        )
        tqdm.write(f"{topic}: {', '.join(closest_topics)}")
