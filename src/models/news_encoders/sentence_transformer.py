from functools import partial

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


class SentenceTransformerNewsEncoder(nn.Module):
    def __init__(self, pretrained_model_name="all-distilroberta-v1", finetune=False):
        super(SentenceTransformerNewsEncoder, self).__init__()
        self.model = SentenceTransformer(pretrained_model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        if not finetune:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def stack_batches(self, news):
        batch_size, n_news, num_words = news["input_ids"].size()
        for key in news:
            news[key] = news[key].reshape(-1, num_words)

        unstack = partial(self.unstack_batches, batch_size=batch_size, n_news=n_news)

        return news, unstack

    @staticmethod
    def unstack_batches(news_vectors, batch_size, n_news):
        return news_vectors.reshape(batch_size, n_news, -1)

    def forward(self, news: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        # batch size, n news, num words
        # or
        # batch size, num words
        titles = news["title"]

        has_multiple_news = titles["input_ids"].dim() == 3
        if has_multiple_news:
            titles, unstack = self.stack_batches(titles)

        for key in titles:
            titles[key] = titles[key].to(self.device)

        news_vectors = self.model(titles)["sentence_embedding"]

        if has_multiple_news:
            news_vectors = unstack(news_vectors)  # type:ignore

        return news_vectors
