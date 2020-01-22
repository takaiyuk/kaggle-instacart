from abc import ABCMeta, abstractmethod
import argparse
import csv
import gc
import inspect
import joblib
import numpy as np
import os
import pandas as pd
import warnings

from .utils import Timer, mkdir, load_yaml

warnings.filterwarnings("ignore")


# https://amalog.hateblo.jp/entry/kaggle-feature-management
class Feature(metaclass=ABCMeta):
    prefix = ""
    suffix = ""
    dir_ = "./feature"

    def __init__(self):
        self.name = self.__class__.__name__
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.train_path = os.path.join(self.dir_, f"{self.name}_train.jbl")
        self.test_path = os.path.join(self.dir_, f"{self.name}_test.jbl")
        self.columns = []
        self.config = load_yaml()

    def run(self):
        with Timer().timer(self.name):
            self.create_features()
            self.drop()
            prefix = self.prefix + "_" if self.prefix else ""
            suffix = "_" + self.suffix if self.suffix else ""
            self.train.columns = prefix + self.train.columns + suffix
            self.test.columns = prefix + self.test.columns + suffix
        return self

    @abstractmethod
    def create_features(self):
        raise NotImplementedError

    def save(self):
        joblib.dump(self.train, self.train_path, compress=3)
        joblib.dump(self.test, self.test_path, compress=3)
        print(f"{self.name} is saved")

    def load(self, columns):
        """columns: 特徴量生成に必要な列名のリスト"""
        trains = []
        tests = []
        for col in columns:
            trains.append(
                joblib.load(
                    os.path.join(self.config["path"]["pdseries"], f"{col}_train.jbl")
                )
            )
        self.train = pd.concat(trains, axis=1)
        for col in columns:
            tests.append(
                joblib.load(
                    os.path.join(self.config["path"]["pdseries"], f"{col}_test.jbl")
                )
            )
        self.test = pd.concat(tests, axis=1)

    def drop(self):
        self.train.drop(self.columns, axis=1, inplace=True)
        self.test.drop(self.columns, axis=1, inplace=True)


# Parse command line argument of whether to overwrite
def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    return parser.parse_args()


# Iterator that instantiates and returns a class that inherits Feature.
def _get_features(namespace):
    for k, v in namespace.items():
        if inspect.isclass(v) and issubclass(v, Feature) and not inspect.isabstract(v):
            yield v()


# Generate features if they do not exist
def generate_features(namespace, overwrite):
    for f in _get_features(namespace):
        if (
            os.path.exists(f.train_path)
            and os.path.exists(f.test_path)
            and not overwrite
        ):
            print(f"{f.name} was skipped")
        else:
            f.run().save()


# Create _features_memo.csv
def create_memo(col_name, desc):

    file_path = os.path.join(Feature.dir, "_features_memo.csv")
    if not os.path.isfile(file_path):
        with open(file_path, "w"):
            pass

    with open(file_path, "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(",")[0] == col_name]
        if len(col) != 0:
            return

        writer = csv.writer(f)
        writer.writerow([col_name, desc])


# Split raw pd.DataFrame to pd.Series column for convenience
def save_column():
    config = load_yaml()
    prefix = config["path"]["pdseries"]
    mkdir(prefix)

    if np.sum([1 if "_train.jbl" in f else 0 for f in os.listdir(prefix)]) == 0:
        print("train prepare")
        train = pd.read_csv(config["path"]["train"])
        for col in train.columns:
            joblib.dump(
                train[col], os.path.join(prefix, f"{col}_train.jbl"), compress=3
            )
        del train
        gc.collect()

    if np.sum([1 if "_test.jbl" in f else 0 for f in os.listdir(prefix)]) == 0:
        print("test prepare")
        test = pd.read_csv(config["path"]["test"])
        for col in test.columns:
            joblib.dump(test[col], os.path.join(prefix, f"{col}_test.jbl"), compress=3)
        del test
        gc.collect()
