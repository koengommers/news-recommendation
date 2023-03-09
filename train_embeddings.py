import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.topic_reads import TopicReadsDataset
from models.skipgram import Skipgram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    epochs: int = 5,
    context_size: int = 1,
    mind_variant: str = "small",
    embedding_dim: int = 100,
    hidden_layer_dim: int = 128,
    batch_size: int = 128,
    learning_rate: float = 0.001,
):
    dataset = TopicReadsDataset(variant=mind_variant, context_size=context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model = Skipgram(
        dataset.number_of_topics,
        dataset.number_of_users,
        embedding_dim,
        hidden_layer_dim,
        context_size,
    ).to(device)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch_num in tqdm(range(epochs)):
        total_train_loss = 0

        for topic, targets in tqdm(dataloader):
            topic = topic.to(torch.long).to(device)
            targets = targets.to(torch.long).to(device)
            model.zero_grad()
            probs = model(topic.squeeze(1))
            loss = loss_function(probs, targets.squeeze(1))
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_train_loss = total_train_loss / len(dataloader)
        print(f"Epochs: {epoch_num + 1} | Train loss: {average_train_loss}")


if __name__ == "__main__":
    typer.run(main)
