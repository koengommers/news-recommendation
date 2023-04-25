import os
import random

from utils.data import get_mind_file, get_mind_path


def get_unique_user_ids():
    user_ids = set()

    train_file = get_mind_file("large", "train", "behaviors.tsv")
    with open(train_file) as f:
        for line in f:
            user_ids.add(line.split()[1])

    dev_file = get_mind_file("large", "dev", "behaviors.tsv")
    with open(dev_file) as f:
        for line in f:
            user_ids.add(line.split()[1])

    return user_ids


def sample_behaviors(sampled_users, variant_name):

    users_in_train = set()
    with open(get_mind_file(variant_name, "train", "behaviors.tsv"), "w") as target:
        with open(get_mind_file("large", "train", "behaviors.tsv")) as source:
            count = 0
            for line in source:
                if line.split()[1] in sampled_users:
                    users_in_train.add(line.split()[1])
                    target.write(line)
                    count += 1
    print(f"Number of users in train: {len(users_in_train)}")
    print(f"Number of logs in train: {count}")

    users_in_dev = set()
    users_in_test = set()
    with open(get_mind_file(variant_name, "dev", "behaviors.tsv"), "w") as target_dev:
        with open(
            get_mind_file(variant_name, "test", "behaviors.tsv"), "w"
        ) as target_test:
            with open(get_mind_file("large", "train", "behaviors.tsv")) as source:
                send_to_test = False
                count_dev = 0
                count_test = 0
                for line in source:
                    if line.split()[1] in sampled_users:
                        if send_to_test:
                            users_in_test.add(line.split()[1])
                            target_test.write(line)
                            count_test += 1
                            send_to_test = False
                        else:
                            target_dev.write(line)
                            users_in_dev.add(line.split()[1])
                            count_dev += 1
                            send_to_test = True
    print(f"Number of users in dev: {len(users_in_dev)}")
    print(f"Number of logs in dev: {count_dev}")
    print(f"Number of users in test: {len(users_in_test)}")
    print(f"Number of logs in test: {count_test}")


def get_unique_news_ids(variant_name, split):
    news_ids = set()
    with open(get_mind_file(variant_name, split, "behaviors.tsv")) as f:
        for line in f:
            ids = line.split()[5:]
            for id in ids:
                news_ids.add(id.split("-")[0])
    return news_ids


def sample_news(variant_name):
    news_ids = get_unique_news_ids(variant_name, "train")
    print(f"Number of news in train: {len(news_ids)}")
    with open(get_mind_file(variant_name, "train", "news.tsv"), "w") as target:
        with open(get_mind_file("large", "train", "news.tsv")) as source:
            for line in source:
                if line.split()[0] in news_ids:
                    target.write(line)

    news_ids = get_unique_news_ids(variant_name, "dev")
    print(f"Number of news in dev: {len(news_ids)}")
    with open(get_mind_file(variant_name, "dev", "news.tsv"), "w") as target:
        with open(get_mind_file("large", "dev", "news.tsv")) as source:
            for line in source:
                if line.split()[0] in news_ids:
                    target.write(line)

    news_ids = get_unique_news_ids(variant_name, "test")
    print(f"Number of news in test: {len(news_ids)}")
    with open(get_mind_file(variant_name, "test", "news.tsv"), "w") as target:
        with open(get_mind_file("large", "dev", "news.tsv")) as source:
            for line in source:
                if line.split()[0] in news_ids:
                    target.write(line)


def sample_data(num_users, variant_name):
    user_ids = get_unique_user_ids()
    sampled_users = set(random.sample(list(user_ids), num_users))

    splits = ["train", "test", "dev"]
    for split in splits:
        new_data_path = get_mind_path(variant_name, split)
        os.makedirs(new_data_path, exist_ok=True)

    sample_behaviors(sampled_users, variant_name)
    sample_news(variant_name)


if __name__ == "__main__":
    sample_data(200000, "200k")
    # TODO: log number of users in each split
