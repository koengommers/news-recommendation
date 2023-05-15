import hydra
import pyrootutils
from omegaconf import DictConfig

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# make linter ignore "Module level import not at top of file"
# ruff: noqa: E402
from src.utils.hydra import print_config


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    print_config(cfg)


if __name__ == "__main__":
    main()
