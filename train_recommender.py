from enum import Enum

import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.behaviors import BehaviorsDataset
from models.BERT_NRMS import BERT_NRMS
from models.NRMS import NRMS
from models.TANR import TANR
from utils.collate import collate_fn
from utils.data import load_pretrained_embeddings
from utils.tokenize import NltkTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Architecture(str, Enum):
    NRMS = "NRMS"
    BERT_NRMS = "BERT-NRMS"
    TANR = "TANR"


class BertPoolingMethod(str, Enum):
    attention = "attention"
    average = "average"
    pooler = "[CLS]"


def main(
    architecture: Architecture,
    epochs: int = 5,
    mind_variant: str = "small",
    batch_size: int = 64,
    negative_sampling_ratio: int = 4,
    num_words_title: int = 20,
    num_words_abstract: int = 50,
    history_length: int = 50,
    learning_rate: float = 0.0001,
    bert_pooling_method: BertPoolingMethod = typer.Option("attention"),
    pretrained_model_name: str = "bert-base-uncased",
    num_batches_show_loss: int = 100,
    use_pretrained_embeddings: bool = True,
    freeze_pretrained_embeddings: bool = False,
):
    # Set up tokenizer
    if architecture == Architecture.BERT_NRMS:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        tokenize = lambda text, length: dict(
            tokenizer(
                text,
                max_length=length,
                padding="max_length",
                truncation=True,
            )
        )
    else:
        tokenizer = NltkTokenizer()
        tokenize = lambda text, length: tokenizer(text, length)

    # Load dataset
    required_news_features = {
        Architecture.NRMS: ["title"],
        Architecture.BERT_NRMS: ["title"],
        Architecture.TANR: ["title", "category"],
    }
    news_features = required_news_features[architecture]
    dataset = BehaviorsDataset(
        mind_variant,
        "train",
        tokenize,
        negative_sampling_ratio,
        num_words_title,
        num_words_abstract,
        history_length,
        news_features,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True
    )

    # Init model
    if architecture == Architecture.NRMS:
        pretrained_embeddings = load_pretrained_embeddings(tokenizer.t2i) if use_pretrained_embeddings else None
        model = NRMS(
            tokenizer.vocab_size + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
        ).to(device)
    elif architecture == Architecture.TANR:
        pretrained_embeddings = load_pretrained_embeddings(tokenizer.t2i) if use_pretrained_embeddings else None
        model = TANR(
            tokenizer.vocab_size + 1,
            dataset.categorical_encoders["category"].n_categories + 1,
            pretrained_embeddings=pretrained_embeddings,
            freeze_pretrained_embeddings=freeze_pretrained_embeddings,
        ).to(device)
    elif architecture == Architecture.BERT_NRMS:
        model = BERT_NRMS(pretrained_model_name, bert_pooling_method.value).to(device)

    # Init optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # Train
    for epoch_num in tqdm(range(1, epochs + 1)):
        total_train_loss = 0.0

        for batch_num, (history, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader)
        ):
            optimizer.zero_grad()

            labels = torch.zeros(batch_size).long().to(device)
            loss = model(candidate_news, history, labels)
            total_train_loss += loss.item()
            loss.backward()

            optimizer.step()

            if batch_num % num_batches_show_loss == 0:
                tqdm.write(
                    f"Loss after {batch_num} batches in epoch {epoch_num}: {total_train_loss / (batch_num)}"
                )

        tqdm.write(
            f"Epochs: {epoch_num} | Average train loss: {total_train_loss / len(dataloader)}"
        )


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
