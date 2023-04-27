from types import SimpleNamespace
import pytest

@pytest.fixture
def dataset():
    return SimpleNamespace(num_words=1000, num_categories=15)
