import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import pandas as pd
import seaborn as sns

from .dataloader import DataLoader
from .const import CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS, TARGET_COLUMN


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
            df_num = self.train.loc[:, self.num_cols].describe().T
            df_num["#null"] = self.train.loc[:, self.num_cols].isnull().sum().values
            df_num["null_rate"] = df_num["#null"] / len(self.train)
            df_num["#uniques"] = self.train.loc[:, self.num_cols].nunique().values
            df_num["nuniques_rate"] = df_num["#uniques"] / len(self.train)
            
            df_cat = pd.DataFrame(index=self.cat_cols)
            df_cat["#null"] = self.train.loc[:, self.cat_cols].isnull().sum().values
            df_cat["null_rate"] = df_cat["#null"] / len(self.train)
            df_cat["#uniques"] = self.train.loc[:, self.cat_cols].nunique().values
            df_cat["nuniques_rate"] = df_cat["#uniques"] / len(self.train)
        else:
            df_num = self.test.loc[:, self.num_cols].describe().T
            df_num["#null"] = self.test.loc[:, self.num_cols].isnull().sum().values
            df_num["null_rate"] = df_num["#null"] / len(self.test)
            df_num["#uniques"] = self.test.loc[:, self.num_cols].nunique().values
            df_num["nuniques_rate"] = df_num["#uniques"] / len(self.test)
            
            df_cat = pd.DataFrame(index=self.cat_cols)
            df_cat["#null"] = self.test.loc[:, self.cat_cols].isnull().sum().values
            df_cat["null_rate"] = df_cat["#null"] / len(self.test)
            df_cat["#uniques"] = self.test.loc[:, self.cat_cols].nunique().values
            df_cat["nuniques_rate"] = df_cat["#uniques"] / len(self.test)
        return df_num, df_cat

    def check_categorical_values(self, is_train: bool = True, N: int = 10) -> dict:
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

    def check_feature_target(self) -> pd.Series:
        return self.train.loc[:, self.num_cols+[TARGET_COLUMN]].corr()[TARGET_COLUMN]

    def check_categorical_target(self) -> dict:
        dfs = {}
        for cat_col in self.cat_cols:
            df_ = self.train.groupby(cat_col)[TARGET_COLUMN].mean()
            dfs[cat_col] = df_
        return dfs

    def check_distplot(self, bins=10) -> None:
        sns.set()
        for num_col in self.num_cols:
            sns.distplot(self.train[num_col], bins=bins, label="train")
            sns.distplot(self.test[num_col], bins=bins, label="test")
            plt.title(num_col)
            plt.legend()
            plt.tight_layout()
            plt.show()

    def check_venn_diagram(self) -> None:
        sns.set()
        for cat_col in self.cat_cols:
            venn2(
                [set(self.train[cat_col]), set(self.test[cat_col])],
                set_labels=("train", "test"),
            )
            plt.title(cat_col)
            plt.tight_layout()
            plt.show()

    def analyze(self):
        print(
            """
        analyzer = DataAnalyzer()
        analyzer.load()
        df_num, df_cat = analyzer.check_statistic()
        dfs = analyzer.check_categorical_values()
        ser = analyzer.check_feature_target()
        dfs = analyzer.check_categorical_target()
        analyzer.check_distplot()
        analyzer.check_venn_diagram()
        """
        )
