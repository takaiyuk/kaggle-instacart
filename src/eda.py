import pandas as pd

from .dataloader import DataLoader
from .const import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS


class DataAnalyzer:
    def __init__(self):
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.dataloader = DataLoader()
        self.cat_cols = CATEGORICAL_COLUMNS
        self.num_cols = NUMERICAL_COLUMNS

    def load(self):
        self.train = self.dataloader.load_train()
        self.test = self.dataloader.load_test()

    def check_statistic(self, is_train: bool = True) -> pd.DataFrame:
        if is_train:
            df = self.train.loc[:, self.num_cols].describe().T
            df["#null"] = self.train.loc[:, self.cat_cols].isnull().sum().values
            df["null_rate"] = df["#null"] / len(self.train)
            df["#uniques"] = self.train.loc[:, self.cat_cols].nunique().values
            df["nuniques_rate"] = df["#uniques"] / len(self.train)
        else:
            df = self.test.loc[:, self.num_cols].describe().T
            df["#null"] = self.train.loc[:, self.cat_cols].isnull().sum().values
            df["null_rate"] = df["#null"] / len(self.train)
            df["#uniques"] = self.train.loc[:, self.cat_cols].nunique().values
            df["nuniques_rate"] = df["#uniques"] / len(self.train)
        return df

    def check_categorical_values(self, N: int = 10, is_train: bool = True) -> dict:
        dfs = {}

        df_ = pd.DataFrame(index=self.cat_cols)
        if is_train:
            df_["#null"] = self.train.loc[:, self.cat_cols].isnull().sum().values
            df_["null_rate"] = df_["#null"] / len(self.train)
            df_["#uniques"] = self.train.loc[:, self.cat_cols].nunique().values
            df_["nuniques_rate"] = df_["#uniques"] / len(self.train)
        else:
            df_["#null"] = self.test.loc[:, self.cat_cols].isnull().sum().values
            df_["null_rate"] = df_["#null"] / len(self.test)
            df_["#uniques"] = self.test.loc[:, self.cat_cols].nunique().values
            df_["nuniques_rate"] = df_["#uniques"] / len(self.test)
        dfs["statics"] = df_

        for cat_col in self.cat_cols:
            if is_train:
                df_ = self.train[cat_col].value_counts().head(N).to_frame()
            else:
                df_ = self.test[cat_col].value_counts().head(N).to_frame()
            dfs[cat_col] = df_
        return dfs

    def check_feature_target(self):
        pass

    def check_categorical_target(self):
        pass

    def check_venn_diagram(self):
        pass
