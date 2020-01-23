import joblib
import pandas as pd


class AbstractDataLoader:
    def __init__(self, config):
        self.config = config
        self.train_path = None
        self.test_path = None
        self.submit_path = None
        if "train" in config["path"].keys():
            self.train_path = config["path"]["train"]
        if "test" in config["path"].keys():
            self.test_path = config["path"]["test"]
        if "sample_submission" in config["path"].keys():
            self.sample_submission_path = config["path"]["sample_submission"]

    def _load(self, path, nrows, parse_dates=None, index_col=None):
        return pd.read_csv(
            path, nrows=nrows, parse_dates=parse_dates, index_col=index_col
        )

    def load_csv(self, path, nrows=None):
        df = self._load(
            path, nrows=nrows, parse_dates=self.config["column"]["timestamp"],
        )
        return df

    def load_train(self, path=None, nrows=None):
        if path is None:
            df = self._load(
                self.train_path,
                nrows=nrows,
                parse_dates=self.config["column"]["timestamp"],
            )
            return df

    def load_test(self, path=None, nrows=None):
        if path is None:
            df = self._load(
                self.test_path,
                nrows=nrows,
                parse_dates=self.config["column"]["timestamp"],
            )
            return df

    def load_sample_submission(self, path=None, nrows=None):
        if path is None:
            return self._load(
                self.sample_submission_path,
                nrows=nrows,
                index_col=self.config["column"]["index"],
            )
