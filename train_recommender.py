import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.behaviors import BehaviorsDataset
from models.NRMS import NRMS
from utils import NltkTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    epochs: int = 5,
    mind_variant: str = "small",
    batch_size: int = 64,
    negative_sampling_ratio: int = 4,
    num_words_title: int = 20,
    history_length: int = 50,
    learning_rate: float = 0.0001,
):
    tokenizer = NltkTokenizer()
    tokenize = lambda text, length: tokenizer.tokenize(text, length)

    dataset = BehaviorsDataset(
        mind_variant,
        "train",
        tokenize,
        negative_sampling_ratio,
        num_words_title,
        history_length,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = NRMS(tokenizer.vocab_size + 1).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch_num in tqdm(range(epochs)):
        total_train_loss = 0

        for batch in tqdm(dataloader):
            candidate_news = batch["candidate_news"].to(device)
            history = batch["history"].to(device)

            optimizer.zero_grad()
            probs = model(candidate_news, history)
            loss = loss_function(probs, torch.zeros(probs.size(0)).long().to(device))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        tqdm.write(
            f"Epochs: {epoch_num + 1} | Average train loss: {total_train_loss / len(dataloader)}"
        )


if __name__ == "__main__":
    typer.run(main)
