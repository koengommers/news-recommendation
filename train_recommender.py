from enum import Enum

import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.behaviors import BehaviorsDataset
from models.BERT_NRMS import BERT_NRMS
from models.NRMS import NRMS
from utils.collate import collate_fn
from utils.tokenize import NltkTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Architecture(str, Enum):
    NRMS = "NRMS"
    BERT_NRMS = "BERT-NRMS"


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
    history_length: int = 50,
    learning_rate: float = 0.0001,
    bert_pooling_method: BertPoolingMethod = typer.Option("attention"),
    pretrained_model_name: str = "bert-base-uncased",
    num_batches_show_loss: int = 100,
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
    dataset = BehaviorsDataset(
        mind_variant,
        "train",
        tokenize,
        negative_sampling_ratio,
        num_words_title,
        history_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Init model
    if architecture == Architecture.NRMS:
        model = NRMS(tokenizer.vocab_size + 1).to(device)
    elif architecture == Architecture.BERT_NRMS:
        model = BERT_NRMS(pretrained_model_name, bert_pooling_method.value).to(device)

    # Init optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch_num in tqdm(range(1, epochs + 1)):
        total_train_loss = 0.0

        for batch_num, (history, candidate_news) in tqdm(
            enumerate(dataloader, 1), total=len(dataloader)
        ):
            optimizer.zero_grad()
            probs = model(candidate_news, history)
            loss = loss_function(probs, torch.zeros(probs.size(0)).long().to(device))
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
