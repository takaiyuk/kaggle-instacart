import os

# PATH ------------------------------
INPUT_PREFIX = "./input/dsb2019"
TRAIN_PATH = os.path.join(INPUT_PREFIX, "train.csv")
TEST_PATH = os.path.join(INPUT_PREFIX, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_PREFIX, "sample_submission.csv")

IMPORTANCE_PREFIX = "./importance"
LOGS_PREFIX = "./logs"
MODEL_PREFIX = "./model"
SUBMIT_PREFIX = "./submit"

# COLUMNS ------------------------------
TARGET_COL = "accuracy"
TARGET_COLUMN = "accuracy"
DROP_COLUMNS = []
NUMERICAL_COLUMNS = ["game_time"]
CATEGORICAL_COLUMNS = ["game_session", "event_code", "title"]
KEY_COLUMN = "installation_id"
VALUE_COLUMN = "title"
TIMESTAMP_COLUMN = "timestamp"
UNUSED_COLUMNS = [""]

# FIT PARAMS ------------------------------
N_FOLDS = 5
KFOLD_METHOD = "normal"  # ["normal", "stratified", "group"]
STRATIFIED_COL = ""
GROUP_COL = ""

# GBM PARAMS ------------------------------
OBJECTIVE = "reg"  # ["reg", "clf"]
GBDT = "lgb"  # [lgb, xgb, cb]
CAT_FEATURES = []

LGB_PARAMS_DEBUG = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "learning_rate": 0.2,
    "num_leaves": 2 ** 2 - 1,
    "seed": 42,
    "verbosity": -1,
}
CB_PARAMS_DEBUG = {}
LGB_PARAMS = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "n_estimators": 1000,
    "learning_rate": 0.2,
    "num_leaves": 2 ** 7 - 1,
    "min_data_in_leaf": 40,
    "seed": 42,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "max_bin": 255,
    "metrics": "mse",
}
CB_PARAMS = {}
ES_ROUNDS = 1000
VERBOSE = 1000
