import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.topic_reads import TopicReadsDataset
from evaluation.embeddings import evaluate_embeddings
from evaluation.utils import print_closest_topics
from models.skipgram import Skipgram

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(version_base=None, config_path="../conf", config_name="train_embeddings")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset = TopicReadsDataset(variant=cfg.data.mind_variant)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = Skipgram(
        dataset.number_of_topics,
        dataset.number_of_users,
        cfg.embedding_dim,
    ).to(device)
    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.learning_rate)

    embeddings = next(model.target_embeddings.parameters()).cpu().data.numpy()
    metrics = evaluate_embeddings(embeddings, dataset)
    metrics_string = " | ".join(
        [f"{key}: {value:.5f}" for key, value in metrics.items()]
    )
    tqdm.write(f"Epochs: 0 | {metrics_string}")
    print_closest_topics(
        embeddings,
        dataset.topic_encoder,
        [
            "movienews",
            "baseball",
            "financenews",
            "newsscience",
            "newsworld",
        ],
    )

    metrics_per_epoch = []

    for epoch_num in tqdm(range(1, cfg.epochs + 1)):
        train_losses = []

        for target, context_positive in tqdm(dataloader):
            current_batch_size = target.size(0)

            target = target.to(torch.long).to(device)
            context_positive = context_positive.to(torch.long).unsqueeze(1)
            context_negative = torch.randint(
                dataset.number_of_users, (current_batch_size, cfg.n_negative_samples)
            )
            context = torch.cat([context_positive, context_negative], dim=1).to(device)

            y_pos = torch.ones(current_batch_size, 1)
            y_neg = torch.zeros(current_batch_size, cfg.n_negative_samples)
            y = torch.cat([y_pos, y_neg], dim=1).to(device)

            model.zero_grad()
            probs = model(target, context)
            loss = loss_function(probs, y)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        embeddings = next(model.target_embeddings.parameters()).cpu().data.numpy()
        metrics = evaluate_embeddings(embeddings, dataset)
        metrics["Train loss"] = np.mean(train_losses)
        metrics["epoch"] = epoch_num
        metrics_per_epoch.append(metrics)
        metrics_string = " | ".join(
            [f"{key}: {value:.5f}" for key, value in metrics.items()]
        )

        tqdm.write(f"Epochs: {epoch_num} | {metrics_string}")
        print_closest_topics(
            embeddings,
            dataset.topic_encoder,
            [
                "movienews",
                "baseball",
                "financenews",
                "newsscience",
                "newsworld",
            ],
        )

        # Save embeddings
        topic_embeddings = pd.DataFrame(
            {
                "subcategory": dataset.topic_encoder.classes_,
                "embedding": list(embeddings),
            }
        )
        topic_embeddings.to_feather(f"topic_embeddings_{epoch_num}.feather")

    pd.DataFrame(metrics_per_epoch).to_csv("metrics.csv", index=False)


if __name__ == "__main__":
    main()
