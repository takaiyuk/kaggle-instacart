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
from typing import Any, Generator, Tuple
import xgboost as xgb


from .const import (
    LGB_PARAMS,
    LGB_PARAMS_DEBUG,
    CB_PARAMS,
    CB_PARAMS_DEBUG,
    ES_ROUNDS,
    VERBOSE,
    CAT_FEATURES,
    OBJECTIVE,
    N_FOLDS,
    KFOLD_METHOD,
    IMPORTANCE_PREFIX,
    STRATIFIED_COL,
    GROUP_COL,
    XGB_PARAMS,
    LINEAR_PARAMS,
    NN_PARAMS,
    MODEL_TYPE,
    TARGET_ENCODING_COLUMNS,
)


def AUC(y_true, y_pred):
    return round(auc(y_true, y_pred), 6)


def MAE(y_true, y_pred):
    return round(mae(y_true, y_pred), 6)


def RMSE(y_true, y_pred):
    return round(mse(y_true, y_pred) ** 0.5, 6)


class ModelParameter:
    def __init__(
        self,
        debug_mode=False,
        lgb_params=LGB_PARAMS,
        cb_params=CB_PARAMS,
        xgb_params=XGB_PARAMS,
        linear_params=LINEAR_PARAMS,
        nn_params=NN_PARAMS,
        es_rounds=ES_ROUNDS,
        verbose=VERBOSE,
        cat_features=CAT_FEATURES,
        objective=OBJECTIVE,
        model_type=MODEL_TYPE,
        n_folds=N_FOLDS,
        kfold_method=KFOLD_METHOD,
        target_encoding_columns=TARGET_ENCODING_COLUMNS,
    ):
        self.lgb_params = lgb_params
        self.cb_params = cb_params
        self.xgb_params = xgb_params
        self.linear_params = linear_params
        self.nn_params = nn_params
        if debug_mode:
            self.lgb_params = LGB_PARAMS_DEBUG
            self.cb_params = CB_PARAMS_DEBUG
        self.early_stopping_rounds = es_rounds
        self.verbose = verbose
        self.cat_features = cat_features
        self.objective = objective
        self.model_type = model_type
        self.n_folds = n_folds
        self.kfold_method = kfold_method
        self.target_encoding_columns = target_encoding_columns


class BaseEstimator:
    def __init__(self, parameter, logger, version):
        self.lgb_params = parameter.lgb_params
        self.cb_params = parameter.cb_params
        self.xgb_params = parameter.xgb_params
        self.linear_params = parameter.linear_params
        self.nn_params = parameter.nn_params
        self.es_rounds = parameter.early_stopping_rounds
        self.verbose = parameter.verbose
        self.cat_features = parameter.cat_features
        self.objective = parameter.objective
        self.model_type = parameter.model_type
        self.n_folds = parameter.n_folds
        self.kfold_method = parameter.kfold_method
        self.target_encoding_columns = parameter.target_encoding_columns
        self.kfolds = ""
        self.model = None
        self.models = []
        self.tr_idxs = []
        self.val_idxs = []
        self.pred_valid = None
        self.pred_test = None
        self.feature_names = []
        self.logger = logger
        self.version = version
        self.fill_values = {}
        self.encoders = []

    def fit_(self):
        raise NotImplementedError

    def predict_(self):
        raise NotImplementedError

    def evaluate_(self, y_true, y_pred):
        if self.objective == "clf":
            return AUC(y_true, y_pred)
        elif self.objective == "reg":
            return RMSE(y_true, y_pred)

    def generate_kfold(
        self, X: pd.DataFrame, y: np.array, shuffle: bool = True
    ) -> Tuple[Generator, pd.DataFrame]:
        self.logger.info(f"n_folds: {self.n_folds}")
        self.logger.info(f"kfold_method: {self.kfold_method}")
        if self.kfold_method == "normal":
            self.kfolds = KFold(self.n_folds, shuffle=shuffle, random_state=42)
            kfold_generator = self.kfolds.split(X, y)
        elif self.kfold_method == "stratified":
            self.logger.info(f"stratified column: {STRATIFIED_COL}")
            self.kfolds = StratifiedKFold(self.n_folds, shuffle=shuffle, random_state=42)
            kfold_generator = self.kfolds.split(X, y)
            if STRATIFIED_COL in X.columns:
                X.drop(STRATIFIED_COL, axis=1, inplace=True)
        elif self.kfold_method == "group":
            self.logger.info(f"group column: {GROUP_COL}")
            self.kfolds = GroupKFold(self.n_folds)
            kfold_generator = self.kfolds.split(X, y, groups=X[GROUP_COL])
            if GROUP_COL in X.columns:
                X.drop(GROUP_COL, axis=1, inplace=True)
        else:
            raise TypeError(
                "generate_kfold required 'kfold_method' must be one of ['normal', 'stratified', 'group']"
            )
        return kfold_generator, X

    def kfold_fit(self, X: pd.DataFrame, y: np.array) -> None:
        self.cat_features = [cat_feature for cat_feature in self.cat_features if cat_feature in X.columns]
        kfold_generator, X = self.generate_kfold(X, y, shuffle=True)
        valid_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kfold_generator):
            self.tr_idxs.append(tr_idx)
            self.val_idxs.append(val_idx)
            X_train, X_valid = X.iloc[tr_idx, :], X.iloc[val_idx, :]
            y_train, y_valid = y[tr_idx], y[val_idx]
            if len(self.target_encoding_columns) > 0:
                print(f"Process target_encoding_columns: {self.target_encoding_columns}")
                X_train["accuracy"] = y_train
                te = TargetEncoder()
                X_train = te.process(X_train, self.target_encoding_columns, is_train=True, target="accuracy", drop=False)
                X_valid = te.process(X_valid, self.target_encoding_columns, is_train=False, target=None, drop=False)
                X_train.drop(["accuracy"], axis=1, inplace=True)
                self.encoders.append(te)
            if self.model_type in ["linear", "nn"]:
                self.fill_values = X_train.mode().iloc[0].to_dict()
                X_train.fillna(self.fill_values, inplace=True)
                X_valid.fillna(self.fill_values, inplace=True)
            print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
            self.fit_(X_train, y_train, X_valid, y_valid)
            print(self.model)
            valid_score = self.evaluate_(y_valid, self.predict_(X_valid))
            valid_scores.append(valid_score)
            self.logger.info(f"fold {fold_idx+1} valid score: {valid_score}")
            del X_train, y_train, X_valid, y_valid
        self.logger.info(f"\nmean valid score: {np.mean(valid_scores)}")

    def kfold_predict(
        self, X_train: pd.DataFrame = None, X_test: pd.DataFrame = None
    ) -> None:
        if X_train is not None:
            self.pred_valid = np.zeros((len(X_train)))
        if X_test is not None:
            self.pred_test = np.zeros((len(X_test)))
            if self.kfold_method == "group":
                X_test.drop(GROUP_COL, axis=1, inplace=True)
        for fold_idx in range(self.kfolds.n_splits):
            val_idx = self.val_idxs[fold_idx]
            self.model = self.models[fold_idx]

            if X_train is not None:
                X_tr = X_train.iloc[val_idx, :]
                if len(self.target_encoding_columns) > 0:
                    X_tr = self.encoders[fold_idx].process(X_tr, self.target_encoding_columns, is_train=False, target=None, drop=False)
                if self.model_type in ["linear", "nn"]:
                    X_tr.fillna(self.fill_values, inplace=True)
                valid_pred = self.predict_(X_tr.values)
                self.pred_valid[val_idx] = valid_pred
            if X_test is not None:
                X_te = X_test.copy()
                if len(self.target_encoding_columns) > 0:
                    X_te = self.encoders[fold_idx].process(X_te, self.target_encoding_columns, is_train=False, target=None, drop=False)
                if self.model_type in ["linear", "nn"]:
                    X_te.fillna(self.fill_values, inplace=True)
                test_pred = self.predict_(X_te.values)
                self.pred_test += test_pred / self.kfolds.n_splits

    def kfold_feature_importance(self, top_features: int = 60) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        for i, model in enumerate(self.models):
            features = self.feature_names.copy()
            if self.model_type == "cb":
                importances = model.get_feature_importance(type="FeatureImportance")
            else:
                importances = model.booster_.feature_importance(importance_type="gain")
            df_tmp = pd.DataFrame(
                {"feature": features, f"importance_{i}": importances}
            ).set_index("feature")
            if i == 0:
                df_fi = df_tmp.copy()
            else:
                df_fi = df_fi.join(df_tmp, how="left", on="feature")
            del df_tmp
        df_fi["importance"] = df_fi.values.mean(axis=1)
        df_fi.sort_values("importance", ascending=False, inplace=True)
        df_fi.reset_index(inplace=True)
        if top_features > 0 and top_features < len(df_fi):
            df_fi = df_fi.iloc[:top_features, :]
        return df_fi

    def plot_feature_importance(self, save: bool = True) -> None:
        df_fi = self.kfold_feature_importance()
        sns.set()
        plt.figure(figsize=(6, 10))
        sns.barplot(y=df_fi["feature"], x=df_fi["importance"])
        plt.tight_layout()
        if save is True:
            plt.savefig(
                f"{IMPORTANCE_PREFIX}/importance_{self.model_type}_{self.version}.png", dpi=150
            )
        else:
            plt.show()
        plt.close()
        return df_fi


class CatboostEstimator(BaseEstimator):
    def __init__(self, parameter, logger, version):
        super().__init__(parameter, logger, version)

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Catboost Params: {self.cb_params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = cb.CatBoostClassifier(**self.cb_params)
        elif self.objective == "reg":
            self.model = cb.CatBoostRegressor(**self.cb_params)
        if X_eval is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_eval, y_eval)],
                early_stopping_rounds=self.es_rounds,
                verbose=self.verbose,
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


class LightgbmEstimator(BaseEstimator):
    def __init__(self, parameter, logger, version):
        super().__init__(parameter, logger, version)

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Lightgbm Params: {self.lgb_params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = lgb.LGBMClassifier(**self.lgb_params)
        elif self.objective == "reg":
            self.model = lgb.LGBMRegressor(**self.lgb_params)
        if X_eval is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_eval, y_eval)],
                early_stopping_rounds=self.es_rounds,
                verbose=self.verbose,
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


class LinearEstimator(BaseEstimator):
    def __init__(self, parameter, logger, version):
        super().__init__(parameter, logger, version)

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Linear Params: {self.linear_params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = LR()
        elif self.objective == "reg":
            self.model = Ridge()
        self.model.fit(X_train, y_train)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            return self.model.predict_proba(X)[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)


class NeuralnetEstimator(BaseEstimator):
    def __init__(self, parameter, logger, version):
        super().__init__(parameter, logger, version)


class XgboostEstimator(BaseEstimator):
    def __init__(self, parameter, logger, version):
        super().__init__(parameter, logger, version)

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"Model Type: {self.model_type}")
        self.logger.info(f"Xgboost Params: {self.xgb_params}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            self.model = xgb.XGBClassifier(**self.xgb_params)
        elif self.objective == "reg":
            self.model = xgb.XGBRegressor(**self.xgb_params)
        if X_eval is not None:
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_eval, y_eval)],
                early_stopping_rounds=self.es_rounds,
                verbose=self.verbose,
            )
            self.logger.info(f"Best Iteration: {self.model.best_iteration}")
        else:
            self.model.fit(X_train, y_train)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            return self.model.predict_proba(X)[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)


class Estimator:
    def __init__(self, model_estimator: BaseEstimator, logger: logging.Loggers, version: int) -> None:
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

            drop_cols = self.X_train.nunique()[self.X_train.nunique() == 1].index.tolist() + self.X_train.isnull().sum()[self.X_train.isnull().sum() == len(self.X_train)].index.tolist()
            if len(drop_cols) != 0:
                self.X_train.drop(drop_cols, axis=1, inplace=True)
                for col in drop_cols:
                    self.selected_columns.remove(col)
            print(self.X_train.shape)
            print(self.X_train.head(1))

            with self.utils.timer("Process Train FeatureSelector"):
                self.selected_features = self.featureselector.select_features(self.X_train.loc[:, self.selected_columns], self.y_train, threshold=NI_THRESHOLD)
                self.logger.info(f"#selected_features: {len(self.selected_features)}")
                self.logger.info(self.selected_features)
                if self.estimator.kfold_method == "stratified" and STRATIFIED_COL not in self.selected_features:
                    use_cols = self.selected_features + [STRATIFIED_COL]
                elif self.estimator.kfold_method == "group" and GROUP_COL not in self.selected_features:
                    use_cols = self.selected_features + [GROUP_COL]
                else:
                    use_cols = self.selected_features
                self.X_train = self.X_train.loc[:, use_cols]

            if self.estimator.model_type in ["linear", "nn"]:
                self.X_train.loc[:, self.selected_features] = self.scaler.process(self.X_train.loc[:, self.selected_features].values, is_train=True)

            self.estimator.kfold_fit(self.X_train, self.y_train)
            self.estimator.kfold_predict(self.X_train)
            valid_score = self.estimator.evaluate_(self.y_train, self.estimator.pred_valid)
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
                X_test.loc[:, self.selected_features] = self.scaler.process(X_test.loc[:, self.selected_features].values, is_train=False)
            print(X_test.shape)
            print(X_test.head(1))

            self.estimator.kfold_predict(X_test=X_test)
