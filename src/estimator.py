import catboost as cb
import gc
import lightgbm as lgb
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.linear_model import Ridge, LogisticRegression as LR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auc
from typing import Generator, Tuple
import xgboost as xgb


class Estimator:
    def __init__(
        self, model_estimator: BaseEstimator, logger: logging.Loggers, version: int
    ) -> None:
        self.selected_columns = []
        self.use_columns = []
        self.selected_features = []
        self.X_train = pd.DataFrame()
        self.y_train = np.zeros(0)
        self.y_true = np.zeros(0)
        self.test_installation_id = np.zeros(0)
        self.logger = logger
        self.version = version
        self.estimator = model_estimator
        self.utils = Utils()
        self.scaler = MyStandardScaler()
        self.featureselector = NullImportance("reg")

    def _select_columns(self, df):
        drop_cols = ["accuracy", "accuracy_group", "installation_id"]
        self.selected_columns = df.drop(drop_cols, axis=1).columns.tolist()
        self.logger.info(f"#selected_columns: {len(self.selected_columns)}")
        print(self.selected_columns)

    def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
        if is_train:
            self._select_columns(df)
            if self.estimator.kfold_method == "stratified":
                use_cols = self.selected_columns + [STRATIFIED_COL]
            elif self.estimator.kfold_method == "group":
                use_cols = self.selected_columns + [GROUP_COL]
            else:
                use_cols = self.selected_columns
            self.X_train, self.y_train = (
                df.loc[:, use_cols],
                df[TARGET_COL].values,
            )
            self.y_true = np.array(df["accuracy_group"].values, dtype=int)
            del df
            gc.collect()

            drop_cols = (
                self.X_train.nunique()[self.X_train.nunique() == 1].index.tolist()
                + self.X_train.isnull()
                .sum()[self.X_train.isnull().sum() == len(self.X_train)]
                .index.tolist()
            )
            if len(drop_cols) != 0:
                self.X_train.drop(drop_cols, axis=1, inplace=True)
                for col in drop_cols:
                    self.selected_columns.remove(col)
            print(self.X_train.shape)
            print(self.X_train.head(1))

            with self.utils.timer("Process Train FeatureSelector"):
                self.selected_features = self.featureselector.select_features(
                    self.X_train.loc[:, self.selected_columns],
                    self.y_train,
                    threshold=NI_THRESHOLD,
                )
                self.logger.info(f"#selected_features: {len(self.selected_features)}")
                self.logger.info(self.selected_features)
                if (
                    self.estimator.kfold_method == "stratified"
                    and STRATIFIED_COL not in self.selected_features
                ):
                    use_cols = self.selected_features + [STRATIFIED_COL]
                elif (
                    self.estimator.kfold_method == "group"
                    and GROUP_COL not in self.selected_features
                ):
                    use_cols = self.selected_features + [GROUP_COL]
                else:
                    use_cols = self.selected_features
                self.X_train = self.X_train.loc[:, use_cols]

            if self.estimator.model_type in ["linear", "nn"]:
                self.X_train.loc[:, self.selected_features] = self.scaler.process(
                    self.X_train.loc[:, self.selected_features].values, is_train=True
                )

            self.estimator.kfold_fit(self.X_train, self.y_train)
            self.estimator.kfold_predict(self.X_train)
            valid_score = self.estimator.evaluate_(
                self.y_train, self.estimator.pred_valid
            )
            self.logger.info(f"\nvalid_score: {valid_score}")
            if self.estimator.model_type in ["cb", "lgb", "xgb"]:
                df_fi = self.estimator.plot_feature_importance()
                self.logger.info(df_fi.loc[:25, ["feature", "importance"]])
            del self.X_train
            gc.collect()

        else:
            if self.estimator.kfold_method == "stratified":
                use_cols = self.selected_features + [STRATIFIED_COL]
            elif self.estimator.kfold_method == "group":
                use_cols = self.selected_features + [GROUP_COL]
            else:
                use_cols = self.selected_features
            if TARGET_COL in df.columns:
                df.drop(TARGET_COL, axis=1, inplace=True)
            X_test = df.loc[:, use_cols]
            self.test_installation_id = df["installation_id"].values
            del df
            gc.collect()
            if self.estimator.model_type == "linear":
                X_test.loc[:, self.selected_features] = self.scaler.process(
                    X_test.loc[:, self.selected_features].values, is_train=False
                )
            print(X_test.shape)
            print(X_test.head(1))

            self.estimator.kfold_predict(X_test=X_test)
