from evaluation.embeddings.analogies import evaluate_with_analogies

def evaluate_embeddings(topic_encoder, embeddings):
    return evaluate_with_analogies(topic_encoder, embeddings)
