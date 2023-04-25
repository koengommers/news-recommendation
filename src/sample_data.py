import os
import random
import time

from utils.data import get_mind_file, get_mind_path


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


def sample_behaviors(sampled_users, variant_name):
    users_in_train = set()
    with open_behaviors_file(variant_name, "train", "w") as target:
        with open_behaviors_file("large", "train") as source:
            count_train = 0
            for line in source:
                if line.split()[1] in sampled_users:
                    users_in_train.add(line.split()[1])
                    target.write(line)
                    count_train += 1
    print(f"Number of users in train: {len(users_in_train)}")
    print(f"Number of logs in train: {count_train}")

    users_in_dev = set()
    users_in_test = set()
    with open_behaviors_file(variant_name, "dev", "w") as target_dev:
        with open_behaviors_file(variant_name, "test", "w") as target_test:
            with open_behaviors_file("large", "train") as source:
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
    print(f"Number of users in dev: {len(users_in_dev)}")
    print(f"Number of logs in dev: {count_dev}")
    print(f"Number of users in test: {len(users_in_test)}")
    print(f"Number of logs in test: {count_test}")


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


def sample_news(variant_name):
    source_splits = ["train", "dev", "dev"]
    target_splits = ["train", "dev", "test"]

    for source_split, target_split in zip(source_splits, target_splits):
        news_ids = get_unique_news_ids(variant_name, target_split)
        print(f"Number of news in {target_split}: {len(news_ids)}")
        copy_news(news_ids, ("large", source_split), (variant_name, target_split))


def sample_data(variant_name, num_users):
    # Step 1: Make directories for new dataset
    splits = ["train", "test", "dev"]
    for split in splits:
        new_data_path = get_mind_path(variant_name, split)
        os.makedirs(new_data_path, exist_ok=True)

    # Step 2: Sample from the user ids
    user_ids = get_unique_user_ids()
    sampled_users = set(random.sample(list(user_ids), num_users))

    # Step 3: Select behaviors from those users
    sample_behaviors(sampled_users, variant_name)

    # Step 4: Copy the news used in those behaviors
    sample_news(variant_name)


if __name__ == "__main__":
    start_time = time.time()
    sample_data("200k", 200000)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Sampled data in {duration:.2f}s")
