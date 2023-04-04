import numpy as np


def mrr(y_true, y_scores):
    sorted = np.flip(np.argsort(y_scores), axis=-1)
    ranks = []
    for i in range(len(y_true)):
        rank = np.argwhere(sorted[i] == y_true[i])[0][0] + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return np.mean(1 / ranks)
