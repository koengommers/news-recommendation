import os
import random
import time

import hydra
import pandas as pd
from omegaconf import DictConfig

from utils.data import get_mind_file, get_mind_path
from utils.hydra import print_config


def open_behaviors_file(variant, split, mode="r"):
    return open(get_mind_file(variant, split, "behaviors.tsv"), mode)


def open_news_file(variant, split, mode="r"):
    return open(get_mind_file(variant, split, "news.tsv"), mode)


def get_unique_user_ids(source_splits=["train", "dev"]):
    user_ids = set()

    for split in source_splits:
        with open_behaviors_file("large", split) as f:
            for line in f:
                user_ids.add(line.split()[1])

    return user_ids


def sample_behaviors(sampled_users, source_variant, variant_name):
    users_in_train = set()
    with open_behaviors_file(variant_name, "train", "w") as target:
        with open_behaviors_file(source_variant, "train") as source:
            count_train = 0
            for line in source:
                if line.split()[1] in sampled_users:
                    users_in_train.add(line.split()[1])
                    target.write(line)
                    count_train += 1

    users_in_dev = set()
    users_in_test = set()
    with open_behaviors_file(variant_name, "dev", "w") as target_dev:
        with open_behaviors_file(variant_name, "test", "w") as target_test:
            with open_behaviors_file(source_variant, "dev") as source:
                count_dev = 0
                count_test = 0
                for line in source:
                    if line.split()[1] in sampled_users:
                        if count_dev > count_test:
                            users_in_test.add(line.split()[1])
                            target_test.write(line)
                            count_test += 1
                        else:
                            target_dev.write(line)
                            users_in_dev.add(line.split()[1])
                            count_dev += 1

    user_counts = {
        "train": len(users_in_train),
        "dev": len(users_in_dev),
        "test": len(users_in_test),
        "total": len(sampled_users),
    }
    log_counts = {
        "train": count_train,
        "dev": count_dev,
        "test": count_test,
        "total": count_train + count_dev + count_test,
    }
    return user_counts, log_counts


def get_unique_news_ids(variant_name, split):
    news_ids = set()
    with open_behaviors_file(variant_name, split) as f:
        for line in f:
            ids = line.split()[5:]
            for id in ids:
                news_ids.add(id.split("-")[0])
    return news_ids


def copy_news(news_ids, source, target):
    source_variant, source_split = source
    target_variant, target_split = target
    with open_news_file(target_variant, target_split, "w") as target:
        with open_news_file(source_variant, source_split) as source:
            for line in source:
                if line.split()[0] in news_ids:
                    target.write(line)


def sample_news(source, variant_name):
    source_splits = ["train", "dev", "dev"]
    target_splits = ["train", "dev", "test"]
    news_counts = {}
    all_news_ids = set()

    for source_split, target_split in zip(source_splits, target_splits):
        news_ids = get_unique_news_ids(variant_name, target_split)
        news_counts[target_split] = len(news_ids)
        all_news_ids.update(news_ids)
        copy_news(news_ids, (source, source_split), (variant_name, target_split))

    news_counts["total"] = len(all_news_ids)

    return news_counts


@hydra.main(version_base=None, config_path="../conf", config_name="sample_data")
def sample_data(cfg: DictConfig):
    random.seed(cfg.data.seed)
    print_config(cfg)

    # Step 1: Make directories for new dataset
    splits = ["train", "test", "dev"]
    for split in splits:
        new_data_path = get_mind_path(cfg.data.target, split)
        os.makedirs(new_data_path, exist_ok=True)

    # Step 2: Sample from the user ids
    print("Sampling users...")
    user_ids = get_unique_user_ids()
    sorted_users = sorted(list(user_ids))  # sort to make it deterministic
    sampled_users = set(random.sample(sorted_users, cfg.data.num_users))

    # Step 3: Select behaviors from those users
    print("Selecting behaviors...")
    user_counts, log_counts = sample_behaviors(
        sampled_users, cfg.data.source, cfg.data.target
    )

    # Step 4: Copy the news used in those behaviors
    print("Copying news...")
    news_counts = sample_news(cfg.data.source, cfg.data.target)

    # Step 5: Stats
    stats = pd.DataFrame(
        [user_counts, log_counts, news_counts], index=["n_users", "n_logs", "n_news"]
    ).transpose()
    print(stats)
    stats.to_csv(os.path.join(get_mind_path(cfg.data.target), "stats.csv"))


if __name__ == "__main__":
    start_time = time.time()
    sample_data()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Sampled data in {duration:.2f}s")
