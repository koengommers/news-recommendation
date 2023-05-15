import os
import sys
import numpy as np
import pytest

src_path = os.path.abspath(os.path.join("./src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from evaluation.metrics import ild

def test_empty_list():
    assert ild([]) == None

def test_single_embedding():
    embedding = np.array([1, 2, 3])
    assert ild([embedding]) == 0

def test_two_identical_embeddings():
    embedding = np.array([1, 2, 3])
    embeddings = [embedding, embedding]
    assert ild(embeddings) == 0

def test_two_different_embeddings():
    embedding_1 = np.array([1, 2, 3])
    embedding_2 = np.array([4, 5, 6])
    embeddings = [embedding_1, embedding_2]
    assert ild(embeddings) == 1 - np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))

def test_three_identical_embeddings():
    embedding = np.array([1, 2, 3])
    embeddings = [embedding, embedding, embedding]
    assert ild(embeddings) == 0

def test_three_different_embeddings():
    embedding_1 = np.array([1, 2, 3])
    embedding_2 = np.array([4, 5, 6])
    embedding_3 = np.array([7, 8, 9])
    embeddings = [embedding_1, embedding_2, embedding_3]
    avg_distance = (1 - np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)) + 
                    1 - np.dot(embedding_1, embedding_3) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_3)) + 
                    1 - np.dot(embedding_2, embedding_3) / (np.linalg.norm(embedding_2) * np.linalg.norm(embedding_3))) / 3
    assert ild(embeddings) == avg_distance

def test_maximum_value():
    embedding_1 = np.array([1, 0, 0])
    embedding_2 = np.array([0, 1, 0])
    embedding_3 = np.array([0, 0, 1])
    embeddings = [embedding_1, embedding_2, embedding_3]
    avg_distance = (1 - np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)) + 
                    1 - np.dot(embedding_1, embedding_3) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_3)) + 
                    1 - np.dot(embedding_2, embedding_3) / (np.linalg.norm(embedding_2) * np.linalg.norm(embedding_3))) / 3
    assert ild(embeddings) == avg_distance == 1

def test_long_embeddings():
    np.random.seed(123)
    num_embeddings = 5
    embedding_length = 100
    embeddings = [np.random.rand(embedding_length) for _ in range(num_embeddings)]
    avg_distance = 0
    for i in range(num_embeddings):
        for j in range(i + 1, num_embeddings):
            avg_distance += (1 - np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
    avg_distance /= num_embeddings * (num_embeddings - 1) / 2
    assert ild(embeddings) == pytest.approx(avg_distance, 1e-10)

def test_invalid_input():
    with pytest.raises(ValueError):
        ild("not a list of embeddings")
