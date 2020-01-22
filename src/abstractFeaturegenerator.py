from abc import ABCMeta, abstractmethod
import argparse
import inspect
import joblib
import pandas as pd
from pathlib import Path

from .utils import Timer


# https://amalog.hateblo.jp/entry/kaggle-feature-management
class Feature(metaclass=ABCMeta):
    prefix = ""
    suffix = ""
    dir_ = "./feature"

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = Path(self.dir_) / f"{self.name}_train.ftr"
        self.test_path = Path(self.dir_) / f"{self.name}_test.ftr"

    def run(self):
        with Timer().time(self.name):
            self.create_features()
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = "_" + self.suffix if self.suffix else ""
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        joblib.dump(self.train, str(self.train_path), compress=3)
        joblib.dump(self.test, str(self.test_path), compress=3)

    def load(self):
        self.train = joblib.load(str(self.train_path))
        self.test = joblib.load(str(self.test_path))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing files"
    )
    return parser.parse_args()


def get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


def generate_features(namespace, overwrite):
    for f in get_features(namespace):
        if f.train_path.exists() and f.test_path.exists() and not overwrite:
            print(f.name, "was skipped")
        else:
            f.run().save()
