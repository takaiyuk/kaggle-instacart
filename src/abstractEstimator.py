import catboost as cb
import gc
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.linear_model import Ridge, LogisticRegression as LR
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auc
from typing import Generator, Tuple

# import xgboost as xgb

from .featureselector import NullImportance
from .utils import Timer


def AUC(y_true, y_pred):
    return round(auc(y_true, y_pred), 6)


def MAE(y_true, y_pred):
    return round(mae(y_true, y_pred), 6)


def RMSE(y_true, y_pred):
    return round(mse(y_true, y_pred) ** 0.5, 6)


class BaseEstimator:
    def __init__(self, config, logger, model_type, version):
        self.config = config
        self.logger = logger
        self.model_type = model_type
        self.version = version
        self.objective = self.config["mode"]["objective"]
        self.kfold_method = self.config["parameter"]["common"]["kfold_method"]
        self.kfold_number = self.config["parameter"]["common"]["kfold_number"]
        self.stratified_column = self.config["column"]["kfold_stratified"]
        self.group_column = self.config["column"]["kfold_group"]
        self.target_column = self.config["column"]["target"]
        self.drop_columns = self.config["column"]["drop"]
        self.importance = self.config["path"]["importance"]
        self.ni_threshold = self.config["parameter"]["null_importance"]["threshold"]

        self.cat_features = []
        self.tr_idxs = []
        self.val_idxs = []
        self.fill_values = {}
        self.model = None
        self.models = []
        self.pred_valid = np.zeros(0)
        self.pred_test = np.zeros(0)

        self.selected_columns = []
        self.selected_features = []
        self.X_train = pd.DataFrame()
        self.y_train = np.zeros(0)
        self.y_true = np.zeros(0)
        self.scaler = MyStandardScaler()
        self.featureselector = NullImportance(self.config["mode"]["objective"])
        self.timer = Timer()

    def fit_(self):
        raise NotImplementedError

    def predict_(self):
        raise NotImplementedError

    def evaluate_(self, y_true, y_pred):
        if self.config["parameter"]["common"]["objective"] == "clf":
            return AUC(y_true, y_pred)
        elif self.config["parameter"]["common"]["objective"] == "reg":
            return RMSE(y_true, y_pred)
        else:
            raise Exception("[Error]: paramater objective must be 'clf' or 'reg'")

    def generate_kfold(
        self, X: pd.DataFrame, y: np.array, shuffle: bool = True
    ) -> Tuple[Generator, pd.DataFrame]:
        if self.kfold_method == "normal":
            self.kfolds = KFold(self.kfold_number, shuffle=shuffle, random_state=42)
            kfold_generator = self.kfolds.split(X, y)
        elif self.kfold_method == "stratified":
            self.logger.info(f"Stratified Column: {self.stratified_column}")
            self.kfolds = StratifiedKFold(
                self.kfold_number, shuffle=shuffle, random_state=42
            )
            kfold_generator = self.kfolds.split(X, y)
            if self.stratified_column in X.columns:
                X.drop(self.stratified_column, axis=1, inplace=True)
        elif self.kfold_method == "group":
            self.logger.info(f"Group Column: {self.group_column}")
            self.kfolds = GroupKFold(self.kfold_number)
            kfold_generator = self.kfolds.split(X, y, groups=X[self.group_column])
            if self.group_column in X.columns:
                X.drop(self.group_column, axis=1, inplace=True)
        else:
            raise TypeError(
                "generate_kfold required 'kfold_method' must be one of ['normal', 'stratified', 'group']"
            )
        return kfold_generator, X

    def kfold_fit(self, X: pd.DataFrame, y: np.array, cat_features: list = []) -> None:
        self.cat_features = cat_features
        kfold_generator, X = self.generate_kfold(X, y, shuffle=True)
        valid_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kfold_generator):
            self.tr_idxs.append(tr_idx)
            self.val_idxs.append(val_idx)
            X_train, X_valid = X.iloc[tr_idx, :], X.iloc[val_idx, :]
            y_train, y_valid = y[tr_idx], y[val_idx]
            if self.model_type in ["linear", "nn"]:
                self.fill_values = X_train.mode().iloc[0].to_dict()
                X_train.fillna(self.fill_values, inplace=True)
                X_valid.fillna(self.fill_values, inplace=True)
            self.features = X_train.columns.tolist()
            print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)

            self.fit_(X_train, y_train, X_valid, y_valid)
            print(self.model)
            valid_score = self.evaluate_(y_valid, self.predict_(X_valid))
            valid_scores.append(valid_score)
            self.logger.info(f"fold {fold_idx+1} valid score: {valid_score}")
            del X_train, y_train, X_valid, y_valid
            gc.collect()
        self.logger.info(f"\nmean valid score: {np.mean(valid_scores)}")

    def kfold_predict(
        self, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None
    ) -> None:
        if X_train is not None:
            self.pred_valid = np.zeros((len(X_train)))
            if self.kfold_method == "group":
                X_train.drop(self.group_column, axis=1, inplace=True)
        if X_test is not None:
            self.pred_test = np.zeros((len(X_test)))
            if self.kfold_method == "group":
                X_test.drop(self.group_column, axis=1, inplace=True)

        for fold_idx in range(self.kfolds.n_splits):
            val_idx = self.val_idxs[fold_idx]
            self.model = self.models[fold_idx]
            if X_train is not None:
                X_val = X_train.iloc[val_idx, :]
                if self.model_type in ["linear", "nn"]:
                    X_val.fillna(self.fill_values, inplace=True)
                self.pred_valid[val_idx] = self.predict_(X_val.values)
            if X_test is not None:
                if self.model_type in ["linear", "nn"]:
                    X_test.fillna(self.fill_values, inplace=True)
                self.pred_test += self.predict_(X_test.values) / self.kfold_number

    def _kfold_feature_importance(self, top_features: int = 60) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        for i, model in enumerate(self.models):
            if self.model_type == "cb":
                features = self.model.get_feature_names()
                importances = model.get_feature_importance(type="FeatureImportance")
            else:
                features = self.model.booster_.feature_name()
                importances = model.booster_.feature_importance(importance_type="gain")
            df_tmp = pd.DataFrame(
                {"feature": features, f"importance_{i}": importances}
            ).set_index("feature")
            if i == 0:
                df_fi = df_tmp.copy()
            else:
                df_fi = df_fi.join(df_tmp, how="left", on="feature")
            del df_tmp
            gc.collect()
        df_fi["importance"] = df_fi.values.mean(axis=1)
        df_fi.sort_values("importance", ascending=False, inplace=True)
        df_fi.reset_index(inplace=True)
        if top_features > 0 and top_features < len(df_fi):
            df_fi = df_fi.iloc[:top_features, :]
        return df_fi

    def plot_feature_importance(self, save: bool = True) -> None:
        df_fi = self._kfold_feature_importance()
        sns.set()
        plt.figure(figsize=(6, 10))
        sns.barplot(y=df_fi["feature"], x=df_fi["importance"])
        plt.tight_layout()
        if save is True:
            plt.savefig(
                f"{self.importance}/importance_{self.model_type}_{self.version}.png",
                dpi=150,
            )
        else:
            plt.show()
        plt.close()
        return df_fi

    def _select_columns(self, df):
        self.selected_columns = df.drop(self.drop_columns, axis=1).columns.tolist()
        self.logger.info(f"#selected_columns: {len(self.selected_columns)}")
        print(self.selected_columns)

    def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
        if is_train:
            self._select_columns(df)
            if self.kfold_method == "stratified":
                use_cols = self.selected_columns + [self.stratified_column]
            elif self.kfold_method == "group":
                use_cols = self.selected_columns + [self.group_column]
            else:
                use_cols = self.selected_columns
            self.X_train, self.y_train = (
                df.loc[:, use_cols],
                df[self.target_column].values,
            )
            self.y_true = np.array(df[self.target_column].values, dtype=int)
            del df
            gc.collect()

            unused_columns = (
                self.X_train.nunique()[self.X_train.nunique() == 1].index.tolist()
                + self.X_train.isnull()
                .sum()[self.X_train.isnull().sum() == len(self.X_train)]
                .index.tolist()
            )
            if len(unused_columns) != 0:
                self.X_train.drop(unused_columns, axis=1, inplace=True)
                for col in unused_columns:
                    self.selected_columns.remove(col)
            print(self.X_train.shape)
            print(self.X_train.head(1))

            with self.timer.timer("Process Train FeatureSelector"):
                self.selected_features = self.featureselector.select_features(
                    self.X_train.loc[:, self.selected_columns],
                    self.y_train,
                    threshold=self.ni_threshold,
                )
                self.logger.info(f"#selected_features: {len(self.selected_features)}")
                self.logger.info(self.selected_features)
                if (
                    self.kfold_method == "stratified"
                    and self.stratified_column not in self.selected_features
                ):
                    use_cols = self.selected_features + [self.stratified_column]
                elif (
                    self.kfold_method == "group"
                    and self.group_column not in self.selected_features
                ):
                    use_cols = self.selected_features + [self.group_column]
                else:
                    use_cols = self.selected_features
                self.X_train = self.X_train.loc[:, use_cols]

            if self.model_type in ["linear", "nn"]:
                self.X_train.loc[:, self.selected_features] = self.scaler.process(
                    self.X_train.loc[:, self.selected_features].values, is_train=True
                )
            self.kfold_fit(self.X_train, self.y_train)
            self.kfold_predict(self.X_train)
            valid_score = self.evaluate_(self.y_train, self.pred_valid)
            self.logger.info(f"\nvalid_score: {valid_score}")
            if self.model_type in ["cb", "lgb", "xgb"]:
                df_fi = self.plot_feature_importance()
                self.logger.info(df_fi.loc[:25, ["feature", "importance"]])

        else:
            if self.kfold_method == "stratified":
                use_cols = self.selected_features + [self.stratified_column]
            elif self.kfold_method == "group":
                use_cols = self.selected_features + [self.group_column]
            else:
                use_cols = self.selected_features
            if self.target_column in df.columns:
                df.drop(self.target_column, axis=1, inplace=True)
            X_test = df.loc[:, use_cols]
            if self.model_type == "linear":
                X_test.loc[:, self.selected_features] = self.scaler.process(
                    X_test.loc[:, self.selected_features].values, is_train=False
                )
            print(X_test.shape)
            print(X_test.head(1))

            self.kfold_predict(X_test=X_test)


class AbstractCatboostEstimator(BaseEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)
        self.params = self.config["parameter"]["cb"]

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Catboost Params: {self.params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = cb.CatBoostClassifier(**self.params)
        elif self.objective == "reg":
            self.model = cb.CatBoostRegressor(**self.params)
        if X_eval is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_eval, y_eval)],
                early_stopping_rounds=self.params["early_stopping_rounds"],
                verbose=self.params["verbose"],
                cat_features=self.cat_features,
            )
            self.logger.info(f"Best Iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X_train, y_train, cat_features=self.cat_features)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            return self.model.predict(X, prediction_type="Probability")[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)


class AbstractLightgbmEstimator(BaseEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)
        self.params = self.config["parameter"]["lgb"]

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Lightgbm Params: {self.params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = lgb.LGBMClassifier(**self.params)
        elif self.objective == "reg":
            self.model = lgb.LGBMRegressor(**self.params)
        if X_eval is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_eval, y_eval)],
                early_stopping_rounds=self.params["early_stopping_rounds"],
                verbose=self.params["verbose"],
                categorical_feature=self.cat_features,
            )
            self.logger.info(f"Best Iteration: {self.model.best_iteration_}")
        else:
            self.model.fit(X_train, y_train, categorical_feature=self.cat_features)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            return self.model.predict_proba(X)[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)


class AbstractLinearEstimator(BaseEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)
        self.params = self.config["parameter"]["linear"]

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Linear Params: {self.params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = LR(**self.params)
        elif self.objective == "reg":
            self.model = Ridge(**self.params)
        self.model.fit(X_train, y_train)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            return self.model.predict_proba(X)[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)


class AbstractNeuralnetEstimator(BaseEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)
        self.params = self.config["parameter"]["nn"]


class AbstractXgboostEstimator(BaseEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)
        self.params = self.config["parameter"]["xgb"]

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        pass

    def predict_(self, X: np.array) -> np.array:
        pass


#     def fit_(
#         self,
#         X_train: pd.DataFrame,
#         y_train: np.array,
#         X_eval: pd.DataFrame = None,
#         y_eval: np.array = None,
#     ) -> None:
#         self.logger.info(f"Model Type: {self.model_type}")
#         self.logger.info(f"Xgboost Params: {self.params}")
#         self.feature_names = X_train.columns.tolist()
#         if self.objective == "clf":
#             self.model = xgb.XGBClassifier(**self.params)
#         elif self.objective == "reg":
#             self.model = xgb.XGBRegressor(**self.params)
#         if X_eval is not None:
#             self.model.fit(
#                 X_train,
#                 y_train,
#                 eval_set=[(X_train, y_train), (X_eval, y_eval)],
#                 early_stopping_rounds=self.params["early_stopping_rounds"],
#                 verbose=self.params["verbose"],
#             )
#             self.logger.info(f"Best Iteration: {self.model.best_iteration}")
#         else:
#             self.model.fit(X_train, y_train)
#         self.models.append(self.model)

#     def predict_(self, X: np.array) -> np.array:
#         if self.objective == "clf":
#             return self.model.predict_proba(X)[:, 1]
#         elif self.objective == "reg":
#             return self.model.predict(X)


class MyStandardScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_(self, X: np.array) -> np.array:
        self.scaler.fit(X)

    def transform_(self, X: np.array) -> np.array:
        return self.scaler.transform(X)

    def process(self, X: np.array, is_train: bool) -> np.array:
        if is_train:
            self.fit_(X)
        return self.transform_(X)
