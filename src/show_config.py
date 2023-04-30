import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="../conf", config_name="train_recommender")
def main(cfg: DictConfig) -> None:
    print("========== Config ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================")


if __name__ == "__main__":
    main()
