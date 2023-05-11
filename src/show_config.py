import hydra
from omegaconf import DictConfig

from utils.hydra import print_config


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    print_config(cfg)


if __name__ == "__main__":
    main()
