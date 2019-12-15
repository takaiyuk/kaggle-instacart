import pandas as pd

from .const import (
    TRAIN_PATH,
    TEST_PATH,
    SAMPLE_SUBMISSION_PATH,
    DROP_COLUMNS,
)


class DataLoader:
    def __init__(self):
        self.tr_path = TRAIN_PATH
        self.te_path = TEST_PATH
        self.sub_path = SAMPLE_SUBMISSION_PATH

    def _load(self, path, nrows, parse_dates=None, index_col=None):
        return pd.read_csv(
            path, nrows=nrows, parse_dates=parse_dates, index_col=index_col
        )

    def load_train(self, path=None, nrows=None):
        if path is None:
            df = self._load(self.tr_path, nrows=nrows, parse_dates=["ts"])
            return df.drop(DROP_COLUMNS, axis=1)

    def load_test(self, path=None, nrows=None):
        if path is None:
            df = self._load(self.te_path, nrows=nrows, parse_dates=["ts"])
            return df.drop(DROP_COLUMNS, axis=1)

    def load_sample_submission(self, path=None, nrows=None):
        if path is None:
            return self._load(self.sub_path, nrows=nrows, index_col="user_id")
