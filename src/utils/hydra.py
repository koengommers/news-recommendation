import os
import time

from omegaconf import DictConfig, OmegaConf


def print_config(cfg: DictConfig) -> None:
    print("========== Config ==========")
    print(OmegaConf.to_yaml(cfg))
    print("============================")


class Run:
    def __init__(self, path):
        self.path = path

    def __repr__(self):
        return f"Run(path={self.path})"

    def get_file_path(self, file_path):
        return os.path.join(self.path, file_path)

    def has_file(self, file_path):
        return os.path.exists(self.get_file_path(file_path))

    def list_files(self, exclude_hydra=True):
        file_paths = []
        for directory, _, files in os.walk(self.path):
            if directory.endswith(".hydra") and exclude_hydra:
                continue

            file_paths.extend(
                [
                    os.path.join(directory, file).replace(self.path, "")[1:]
                    for file in files
                ]
            )
        return file_paths

    def _load_yaml(self, path):
        return OmegaConf.load(self.get_file_path(path))

    @property
    def archived(self):
        return self.has_file(".archived")

    def archive(self):
        with open(self.get_file_path(".archived"), "w") as f:
            f.write(f"{int(time.time())}")

    def unarchive(self):
        os.remove(self.get_file_path(".archived"))

    @property
    def config(self):
        return self._load_yaml(".hydra/config.yaml")

    def select_config_param(self, path):
        return OmegaConf.select(self.config, path)

    def show_config(self):
        print(OmegaConf.to_yaml(self.config))

    @property
    def overrides(self):
        raw = self._load_yaml(".hydra/overrides.yaml")
        overrides = {}
        for item in raw:
            if not isinstance(item, str):
                continue
            splitted = item.split("=")
            key = splitted[0]
            value = len(splitted) == 1 or splitted[1]
            overrides[key] = value
        return overrides

    @property
    def hydra(self):
        return self._load_yaml(".hydra/hydra.yaml").hydra

    def __hash__(self):
        return hash(self.path)

    def __lt__(self, obj):
        return self.path < obj.path

    def __gt__(self, obj):
        return self.path > obj.path

    def __le__(self, obj):
        return self.path <= obj.path

    def __ge__(self, obj):
        return self.path >= obj.path

    def __eq__(self, obj):
        return self.path == obj.path


class RunCollection(list):
    def __init__(self, runs):
        super().__init__(runs)

    @classmethod
    def from_path(cls, path, include_archived=False):
        runs = []
        for directory, subdirectories, files in os.walk(path):
            is_hydra_run = ".hydra" in subdirectories
            is_archived = ".archived" in files
            skip_because_archived = is_archived and not include_archived
            if is_hydra_run and not skip_because_archived:
                runs.append(Run(directory))
        return cls(runs)

    def filter(self, fn):
        return self.__class__(filter(fn, self))

    def filter_by_job(self, job_name):
        return self.filter(lambda run: run.hydra.job.name == job_name)

    def filter_by_override(self, name, value):
        return self.filter(lambda run: run.overrides.get(name) == value)

    def filter_by_config_param(self, path: str):
        return self.filter(lambda run: run.select_config_param(path) is not None)

    def filter_by_config_value(self, path: str, value):
        return self.filter(lambda run: run.select_config_param(path) == value)

    def sort(self, key, reverse=False):
        return self.__class__(sorted(self, key=key, reverse=reverse))

    def sort_by_config_value(self, path, reverse=False):
        return self.sort(lambda run: run.select_config_param(path), reverse=reverse)

    def one(self):
        assert len(self) == 1
        return self[0]
