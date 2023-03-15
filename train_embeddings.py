import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.topic_reads import TopicReadsDataset
from evaluate_embeddings import evaluate_embeddings, find_closest_embedding
from models.skipgram import Skipgram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    epochs: int = 5,
    context_size: int = 1,
    mind_variant: str = "small",
    embedding_dim: int = 100,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    n_negative_samples: int = 4,
):
    dataset = TopicReadsDataset(variant=mind_variant, context_size=context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Skipgram(
        dataset.number_of_topics,
        dataset.number_of_users,
        embedding_dim,
    ).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    embeddings = next(model.target_embeddings.parameters()).cpu().data.numpy()
    tqdm.write("== Closest topics ==")
    for topic in [
        "movienews",
        "baseball",
        "financenews",
        "newsscience",
        "newsworld",
    ]:
        closest_topics = find_closest_embedding(
            embeddings, dataset.topic_encoder, topic
        )
        print(f"{topic}: {', '.join(closest_topics)}")

    for epoch_num in tqdm(range(epochs)):
        total_train_loss = 0

        for target, context_positive in tqdm(dataloader):
            batch_size = target.size(0)

            target = target.to(torch.long).to(device)
            context_positive = context_positive.to(torch.long)
            context_negative = torch.randint(dataset.number_of_users, (batch_size, n_negative_samples))
            context = torch.cat([context_positive, context_negative], dim=1).to(device)

            y_pos = torch.ones(batch_size, 1)
            y_neg = torch.zeros(batch_size, n_negative_samples)
            y = torch.cat([y_pos, y_neg], dim=1).to(device)

            model.zero_grad()
            probs = model(target.squeeze(1), context)
            loss = loss_function(probs, y)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()

        embeddings = next(model.target_embeddings.parameters()).cpu().data.numpy()
        metrics = evaluate_embeddings(dataset.topic_encoder, embeddings)

        average_train_loss = total_train_loss / len(dataloader)
        tqdm.write(
            f"Epochs: {epoch_num + 1} | Train loss: {average_train_loss} | P@1: {metrics['P@1']} | P@5: {metrics['P@5']} | MRR: {metrics['MRR']:.5f}"
        )

        tqdm.write("== Closest topics ==")
        for topic in [
            "movienews",
            "baseball",
            "financenews",
            "newsscience",
            "newsworld",
        ]:
            closest_topics = find_closest_embedding(
                embeddings, dataset.topic_encoder, topic
            )
            tqdm.write(f"{topic}: {', '.join(closest_topics)}")


if __name__ == "__main__":
    typer.run(main)
