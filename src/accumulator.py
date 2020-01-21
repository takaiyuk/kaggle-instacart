from .utils import load_yaml


class Accumulator:
    def __init__(self):
        self.config = load_yaml()

        self.num_cols = self.config["column"]["categorical"]
        self.cat_cols = self.config["column"]["numerical"]
        self.target_col = self.config["column"]["target"]
        self.key_col = self.config["column"]["key"]
        self.ts_col = self.config["column"]["timestamp"]
