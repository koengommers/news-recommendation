import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.behaviors import BehaviorsDataset
from utils import NltkTokenizer


def main(
    epochs: int = 5,
    mind_variant: str = "small",
    batch_size: int = 64,
    negative_sampling_ratio: int = 4,
    num_words_title: int = 60,
    history_length: int = 50,
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

    for epoch_num in tqdm(range(epochs)):
        for item in dataloader:
            print(item)
            exit()


if __name__ == "__main__":
    typer.run(main)
