import os

# PATH ------------------------------
INPUT_PREFIX = "./input/nikkei"
TRAIN_PATH = os.path.join(INPUT_PREFIX, "train.csv")
TEST_PATH = os.path.join(INPUT_PREFIX, "test.csv")
SAMPLE_SUBMISSION_PATH = os.path.join(INPUT_PREFIX, "sample_submission.csv")

IMPORTANCE_PREFIX = "./importance"
LOGS_PREFIX = "./logs"
MODEL_PREFIX = "./model"
SUBMIT_PREFIX = "./submit"

# COLUMNS ------------------------------
TARGET_COLUMN = "age"
DROP_COLUMNS = [
    "er_dev_browser_family",
    "er_dev_browser_version",
    "er_rfs_service_name",
    "er_rfs_service_type",
    "er_rfc_kiji_id_raw",
]
NUMERICAL_COLUMNS = ["ig_ctx_red_viewed_percent", "ig_ctx_red_elapsed_since_page_load"]
CATEGORICAL_COLUMNS = [
    "er_geo_bc_flag",
    "ig_ctx_product",
    "er_geo_pref_j_name",
    "er_geo_city_j_name",
    "er_geo_country_code",
    "er_dev_device_name",
    "er_dev_device_type",
    "er_dev_manufacture",
    "er_dev_os_family",
    "er_dev_os_version",
    "er_rfs_reffered_visit",
    "ig_usr_connection",
]
KEY_COLUMN = "user_id"
VALUE_COLUMN = "kiji_id"
TIMESTAMP_COLUMN = "ts"
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
