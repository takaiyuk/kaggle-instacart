import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


class NullImportance:
    def __init__(self, objective):
        assert objective in ["reg", "clf"]
        self.clf = None
        self.objective = objective
        self.actual_imp_df = pd.DataFrame()
        self.null_imp_df = pd.DataFrame()

    def _fit(self, X: pd.DataFrame, y: np.array, is_shuffle: bool) -> None:
        if is_shuffle:
            y = np.random.permutation(y)
        if self.objective == "reg":
            self.clf = lgb.LGBMRegressor(random_state=42)
        else:
            self.clf = lgb.LGBMClassifier(random_state=42)
        self.clf.fit(X, y)

    def _feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        imp_df = pd.DataFrame()
        imp_df["feature"] = X.columns
        imp_df["importance"] = self.clf.feature_importances_
        imp_df.sort_values("importance", ascending=False, inplace=True)
        return imp_df

    def get_feature_importances(
        self, X: pd.DataFrame, y: np.array, is_shuffle: bool = False
    ) -> pd.DataFrame:
        self._fit(X, y, is_shuffle)
        return self._feature_importance(X)

    def get_null_importance(
        self, X: pd.DataFrame, y: np.array, is_shuffle: bool = True
    ) -> pd.DataFrame:
        N_RUNS = 100
        null_imp_df = pd.DataFrame()
        for i in tqdm(range(N_RUNS)):
            imp_df = self.get_feature_importances(X, y, is_shuffle)
            imp_df["run"] = i + 1
            null_imp_df = pd.concat([null_imp_df, imp_df], ignore_index=True)
        return null_imp_df

    def select_features(
        self, X: pd.DataFrame = None, y: np.array = None, threshold: int = 80
    ) -> list:
        # 閾値を超える特徴量を取得
        if len(self.actual_imp_df) == 0:
            self.actual_imp_df = self.get_feature_importances(X, y)
        if len(self.null_imp_df) == 0:
            self.null_imp_df = self.get_null_importance(X, y)

        imp_features = []
        for feature in self.actual_imp_df["feature"]:
            actual_value = self.actual_imp_df.query(f"feature=='{feature}'")[
                "importance"
            ].values
            null_value = self.null_imp_df.query(f"feature=='{feature}'")[
                "importance"
            ].values
            percentage = (null_value < actual_value).sum() / null_value.size * 100
            if percentage >= threshold:
                imp_features.append(feature)
        return imp_features

    def _display_distribution(self, feature: str) -> None:
        # ある特徴量に対する重要度を取得
        actual_imp = self.actual_imp_df.query(f"feature == '{feature}'")[
            "importance"
        ].mean()
        null_imp = self.null_imp_df.query(f"feature == '{feature}'")["importance"]

        # 可視化
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        a = ax.hist(null_imp, label="Null importances")
        ax.vlines(
            x=actual_imp,
            ymin=0,
            ymax=np.max(a[0]),
            color="r",
            linewidth=10,
            label="Real Target",
        )
        ax.legend(loc="upper right")
        ax.set_title(f"Importance of {feature.upper()}", fontweight="bold")
        plt.xlabel(f"Null Importance Distribution for {feature.upper()}")
        plt.ylabel("Importance")
        plt.show()
        plt.close()

    def display_distributions(
        self, X: pd.DataFrame = None, y: np.array = None,
    ) -> None:
        if len(self.actual_imp_df) == 0:
            self.actual_imp_df = self.get_feature_importances(X, y)
        if len(self.null_imp_df) == 0:
            self.null_imp_df = self.get_null_importance(X, y)
        for feature in self.actual_imp_df["feature"][:5]:
            self._display_distribution(feature)


def test_null_importance():
    from sklearn.datasets import load_boston

    boston = load_boston()
    X = boston["data"]
    y = boston["target"]
    features = boston["feature_names"]
    X_ = pd.DataFrame(X, columns=features)
    ni = NullImportance("reg")
    # ni.display_distributions(X_, y)
    selected_features = ni.select_features(X_, y)
    print(selected_features)
