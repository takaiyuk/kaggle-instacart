from contextlib import contextmanager
import datetime
import joblib
import logging
import numpy as np
import os
import pandas as pd
import time
from typing import Any, Tuple
import yaml


class Logger:
    def __init__(self, log_dir: str = "./logs") -> None:
        self.log_dir = log_dir
        self.version = None

    def _make_new_version(self) -> None:
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        logs = os.listdir(self.log_dir)
        logs = [int(log.replace(".log", "")) for log in logs if ".log" in log]

        if logs == []:
            new_version = 0
        else:
            new_version = np.max(logs) + 1

        if len(str(new_version)) == 1:
            new_version = "0" + str(new_version)
        else:
            new_version = str(new_version)

        self.version = new_version

    def _make_logs_dir(self) -> None:
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.mkdir(self.log_dir)

    # print() の代わりに logger.info() を使用することでログ出力可能
    def get_logger(self) -> Tuple[logging.Logger, int]:
        self._make_logs_dir()
        self._make_new_version()
        logger_ = logging.getLogger("main")
        logger_.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f"{self.log_dir}/{self.version}.log")
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger_.addHandler(fh)
        logger_.addHandler(ch)
        return logger_, self.version


class Timer:
    def __init__(self):
        self

    @contextmanager
    def timer(self, name):
        t0 = time.time()
        print(f"[{name}] start")
        yield
        print(f"[{name}] done in {time.time() - t0:.1f} s")

    def start(self, fname="", logger=None):
        self.st_time = time.time()
        if logger is None:
            print(f"START {fname} time: {datetime.datetime.today()}")
        else:
            logger.info(f"START {fname} time: {datetime.datetime.today()}")

    def _elapesd_minutes(self):
        return (time.time() - self.st_time) / 60

    def end(self, fname="", logger=None):
        if logger is None:
            print(f"FINISH {fname} time: {self._elapesd_minutes():.1f}min.")
        else:
            logger.info(f"FINISH {fname} time: {self._elapesd_minutes():.1f}min.")


def load_joblib(path: str) -> Any:
    return joblib.load(path)


def load_yaml(path: str = "./config.yml") -> dict:
    with open(path, "r") as f:
        return yaml.load(f)


def mkdir(path) -> None:
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """ iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024 ** 2
    numerics = [
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
    ]  # https://stackoverflow.com/questions/56640472/downcast-to-float16-in-pandas-to-numeric)

    for col in df.columns:
        col_type = df[col].dtype
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose is True:
        print(
            "Memory usage after optimization is: {:.2f} MB (Decreased by {:.1f}%)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def save_joblib(target: Any, path: str) -> None:
    joblib.dump(target, path, compress=3)
