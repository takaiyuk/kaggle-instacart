import catboost as cb
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from typing import Generator


from .const import (
    LGB_PARAMS,
    LGB_PARAMS_DEBUG,
    CB_PARAMS,
    CB_PARAMS_DEBUG,
    ES_ROUNDS,
    VERBOSE,
    CAT_FEATURES,
    OBJECTIVE,
    GBDT,
    N_FOLDS,
    KFOLD_METHOD,
    IMPORTANCE_PREFIX,
    STRATIFIED_COL,
    GROUP_COL,
)


def RMSE(y_true, y_pred):
    return round(mse(y_true, y_pred) ** 0.5, 6)


def MAE(y_true, y_pred):
    return round(mae(y_true, y_pred) ** 0.5, 6)


class LightgbmParameter:
    def __init__(
        self,
        debug_mode=False,
        lgb_params=LGB_PARAMS,
        cb_params=CB_PARAMS,
        es_rounds=ES_ROUNDS,
        verbose=VERBOSE,
        cat_features=CAT_FEATURES,
        objective=OBJECTIVE,
        gbdt=GBDT,
    ):
        self.lgb_params = lgb_params
        self.cb_params = cb_params
        if debug_mode:
            self.lgb_params = LGB_PARAMS_DEBUG
            self.cb_params = CB_PARAMS_DEBUG
        self.early_stopping_rounds = es_rounds
        self.verbose = verbose
        self.cat_features = cat_features
        self.objective = objective
        self.gbdt = gbdt


class Estimator:
    def __init__(self, parameter, logger, version):
        self.lgb_params = parameter.lgb_params
        self.cb_params = parameter.cb_params
        self.es_rounds = parameter.early_stopping_rounds
        self.verbose = parameter.verbose
        self.cat_features = parameter.cat_features
        self.objective = parameter.objective
        self.gbdt = parameter.gbdt
        self.n_folds = parameter.n_folds
        self.kfold_method = parameter.kfold_method
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

    def fit_(
        self,
        X_train: pd.DataFrame,
        y_train: np.array,
        X_eval: pd.DataFrame = None,
        y_eval: np.array = None,
    ) -> None:
        self.logger.info(f"gbdt: {self.gbdt}")
        self.feature_names = X_train.columns.tolist()
        if self.objective == "clf":
            if self.gbdt == "lgb":
                self.model = lgb.LGBMClassifier(**self.lgb_params)
            elif self.gbdt == "cb":
                self.model = cb.CatBoostClassifier(**self.cb_params)
        elif self.objective == "reg":
            if self.gbdt == "lgb":
                self.model = lgb.LGBMRegressor(**self.lgb_params)
            elif self.gbdt == "cb":
                self.model = cb.CatBoostRegressor(**self.cb_params)
        if X_eval is not None:
            if self.gbdt == "lgb":
                self.logger.info(f"lgb params: {self.lgb_params}")
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_eval, y_eval)],
                    early_stopping_rounds=self.es_rounds,
                    verbose=self.verbose,
                    categorical_feature=self.cat_features,
                )
            elif self.gbdt == "cb":
                self.logger.info(f"catboost params: {self.cb_params}")
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_eval, y_eval)],
                    early_stopping_rounds=self.es_rounds,
                    verbose=self.verbose,
                    cat_features=self.cat_features,
                )
            self.logger.info(f"early stopping rounds: {self.es_rounds}")
        else:
            if self.gbdt == "lgb":
                self.logger.info(f"lgb params: {self.lgb_params}")
                self.model.fit(X_train, y_train, categorical_feature=self.cat_features)
            elif self.gbdt == "cb":
                self.logger.info(f"catboost params: {self.cb_params}")
                self.model.fit(X_train, y_train, cat_features=self.cat_features)
        self.models.append(self.model)

    def predict_(self, X: np.array) -> np.array:
        if self.objective == "clf":
            if self.gbdt == "lgb":
                return self.model.predict_proba(X)[:, 1]
            elif self.gbdt == "cb":
                return self.model.predict(X, prediction_type="Probability")[:, 1]
        elif self.objective == "reg":
            return self.model.predict(X)

    def evaluate_(self, y_true, y_pred):
        return RMSE(y_true, y_pred)

    def generate_kfold(
        self, X, y, kfold_method: str, n_splits: int = N_FOLDS, shuffle: bool = True
    ) -> Generator:
        self.logger.info(f"n_folds: {self.n_folds}")
        self.logger.info(f"kfold_method: {self.kfold_method}")
        if kfold_method == "normal":
            self.kfolds = KFold(n_splits, shuffle=shuffle, random_state=42)
            kfold_generator = self.kfolds.split(X, y)
        elif kfold_method == "stratified":
            self.logger.info(f"stratified column: {STRATIFIED_COL}")
            self.kfolds = StratifiedKFold(n_splits, shuffle=shuffle, random_state=42)
            kfold_generator = self.kfolds.split(X, y)
            if STRATIFIED_COL in X.columns:
                X.drop(STRATIFIED_COL, axis=1, inplace=True)
        elif kfold_method == "group":
            self.logger.info(f"group column: {GROUP_COL}")
            self.kfolds = GroupKFold(n_splits)
            kfold_generator = self.kfolds.split(X, y, groups=X[GROUP_COL])
            X.drop(GROUP_COL, axis=1, inplace=True)
        else:
            raise TypeError(
                "generate_kfold() required 'kfold_method' must be one of ['normal', 'stratified', 'group']"
            )
        return kfold_generator

    def kfold_fit(self, X: pd.DataFrame, y: np.array) -> None:
        kfold_generator = self.generate_kfold(X, y, KFOLD_METHOD)
        valid_scores = []
        for fold_idx, (tr_idx, val_idx) in enumerate(kfold_generator):
            self.tr_idxs.append(tr_idx)
            self.val_idxs.append(val_idx)
            X_train, X_valid = X.iloc[tr_idx, :], y[tr_idx].values
            y_train, y_valid = X.iloc[val_idx, :], y[val_idx].values
            print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
            self.fit_(X_train, y_train, X_valid, y_valid)
            print(self.model)
            valid_score = self.evaluate_(y_valid, self.predict_(X_valid))
            valid_scores.append(valid_score)
            if self.objective == "clf":
                self.logger.info(f"fold {fold_idx+1} valid_auc: {valid_score}")
            elif self.objective == "reg":
                self.logger.info(f"fold {fold_idx+1} valid_rmse: {valid_score}")
            del X_train, y_train, X_valid, y_valid
        self.logger.info(f"\nmean eval score: {np.mean(valid_scores)}")

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
                valid_pred = self.predict_(X_tr.values)
                self.pred_valid[val_idx] = valid_pred
            if X_test is not None:
                X_te = X_test.copy()
                test_pred = self.predict_(X_te.values)
                self.pred_test += test_pred / self.kfolds.n_splits

    def kfold_feature_importance(self, top_features: int = 60) -> pd.DataFrame:
        df_fi = pd.DataFrame()
        for i, model in enumerate(self.models):
            features = self.feature_names.copy()
            importances = model.feature_importances_
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
        if top_features > 0:
            df_fi = df_fi.iloc[:top_features, :]
        df_fi.reset_index(inplace=True)
        return df_fi

    def plot_feature_importance(self, save: bool = True) -> None:
        df_fi = self.kfold_feature_importance()
        sns.set()
        plt.figure(figsize=(6, 10))
        sns.barplot(y=df_fi["feature"], x=df_fi["importance"])
        plt.tight_layout()
        if save is True:
            plt.savefig(
                f"{IMPORTANCE_PREFIX}/importance_{self.gbdt}_{self.version}.png"
            )
        else:
            plt.show()
        plt.close()
        return df_fi
