from omegaconf import DictConfig, OmegaConf

def print_config(cfg: DictConfig) -> None:
    print("========== Config ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================")
