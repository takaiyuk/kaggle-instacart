import category_encoders as ce
import numpy as np
import pandas as pd
from scipy.stats import mode

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .const import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    KEY_COLUMN,
    VALUE_COLUMN,
    TIMESTAMP_COLUMN,
)


def null_count(ser):
    return len(ser) - ser.count()


def mod(ser):
    return mode(ser)[0][0]


class BaseFitTransformer:
    """
    http://contrib.scikit-learn.org/categorical-encoding/index.html
    Base class for categorical encoder classes. All other classes should implement fit_ and transform_ methods.

    The way to use is like below:
    ```
    he = HogeEncoder()
    train_ = he.process(train, columns, is_train=True)
    test_ = he.process(test, columns, is_train=False)
    ```
    """

    def __init__(self):
        self.encoder = None
        self.tfv = None
        self.svd = None

    def fit_(self):
        pass

    def transform_(self):
        pass

    def process(
        self,
        df: pd.DataFrame,
        columns: list,
        is_train: bool,
        target: str = None,
        drop: bool = False,
    ):
        if is_train:
            self.fit_(df, columns, target)
        df_ = self.transform_(df, columns)
        if drop:
            df.drop(columns, axis=1, inplace=True)
        if df_.index.name is None:
            df = pd.concat((df, df_), axis=1)
        else:
            df = df.join(df_, how="left", on=df_.index.name)
        return df


class OrdinalEncoder(BaseFitTransformer):
    """http://contrib.scikit-learn.org/categorical-encoding/ordinal.html"""

    def __init__(self):
        super().__init__()

    def fit_(self, df: pd.DataFrame, columns: list, target: str):
        self.encoder = ce.OrdinalEncoder(
            cols=columns, handle_unknown="value", handle_missing="value"
        )
        self.encoder.fit(df.loc[:, columns])

    def transform_(self, df: pd.DataFrame, columns: list):
        return self.encoder.transform(df.loc[:, columns]).add_suffix("_encoded")


class TargetEncoder(BaseFitTransformer):
    """http://contrib.scikit-learn.org/categorical-encoding/catboost.html"""

    def __init__(self):
        super().__init__()

    def fit_(self, df: pd.DataFrame, columns: list, target: str):
        self.encoder = ce.cat_boost.CatBoostEncoder(
            cols=columns, handle_unknown="value", handle_missing="value"
        )
        self.encoder.fit(df.loc[:, columns], df[target])

    def transform_(self, df: pd.DataFrame, columns: list):
        return self.encoder.transform(df.loc[:, columns])


class CategoricalAggregator:
    def __init__(self, agg_methods=["count", "nunique", null_count, mod]):
        self.agg_methods = agg_methods

    def process(self, df, columns, key_column, drop=False):
        for col in columns:
            df_agg = df.groupby(key_column).agg({col: self.agg_methods})
            df_agg.columns = [f"{col[0]}_{col[1]}" for col in df_agg.columns]
            df = df.join(df_agg, how="left", on=key_column)
        if drop:
            df.drop(columns, axis=1, inplace=True)
        return df


class NumericalAggregator:
    def __init__(self):
        pass

    def process(self, df, columns, key_column, drop=False):
        for col in columns:
            df_agg = df.groupby(key_column)[col].describe()
            df_agg.columns = [
                f'{col}_{col_agg.replace("%", "pct")}' for col_agg in df_agg.columns
            ]
            df = df.join(df_agg, how="left", on=key_column)
        if drop:
            df.drop(columns, axis=1, inplace=True)
        return df


class TimestampAggregator:
    def __init__(self):
        pass

    def _calc_latest(self, df, ts_column, latest_column, key_column):
        df[latest_column] = df.groupby(key_column)[ts_column].transform("max")
        return df

    def preprocess(self, df, ts_column, latest_column="latest", key_column=None):
        df[ts_column] = pd.to_datetime(df[ts_column])
        df["year"] = df[ts_column].dt.year
        df["month"] = df[ts_column].dt.month
        df["day"] = df[ts_column].dt.day
        df["dayofweek"] = df[ts_column].dt.dayofweek
        df["is_weekend"] = (df[ts_column].dt.weekday >= 5).astype(int)
        df["hour"] = df[ts_column].dt.hour
        df["minute"] = df[ts_column].dt.minute
        df["second"] = df[ts_column].dt.second

        if latest_column not in df.columns:
            df = self._calc_latest(df, ts_column, latest_column, key_column)
        df[latest_column] = pd.to_datetime(df[latest_column])
        df["year_latest"] = df[latest_column].dt.year
        df["month_latest"] = df[latest_column].dt.month
        df["day_latest"] = df[latest_column].dt.day
        df["dayofweek_latest"] = df[latest_column].dt.dayofweek
        df["is_weekend_latest"] = (df[latest_column].dt.weekday >= 5).astype(int)
        df["hour_latest"] = df[latest_column].dt.hour
        df["minute_latest"] = df[latest_column].dt.minute
        df["second_latest"] = df[latest_column].dt.second

        return df


class BaseFilterAggregator:
    def __init__(self, user_id, ts_column, latest_column=None):
        self.user_id = user_id
        self.ts_column = ts_column
        self.latest_column = latest_column

    def _diff(self, df):
        df["ts_diff_days"] = (df[self.latest_column] - df[self.ts_column]).apply(
            lambda x: x.days
        )
        return df

    def _filter(self, df, filter_col, window):
        df = self._diff(df)
        return df[df[filter_col] <= window]

    def process(self, df, window):
        pass


class LastNDayAggregator(BaseFilterAggregator):
    def __init__(self, user_id, ts_column, latest_column=None):
        super().__init__(user_id, ts_column, latest_column)
        self.aggregator = NumericalAggregator()

    def preprocess(self, df, filter_col, window, numrical_columns):
        df = self._filter(df, filter_col, window)
        df = self.aggregator.process(df, numrical_columns, self.user_id)
        return df


class LastNSessionAggregator(BaseFilterAggregator):
    def __init__(self, user_id, ts_column, latest_column=None):
        super().__init__(user_id, ts_column, latest_column)
        self.aggregator = NumericalAggregator()

    def preprocess(self, df, filter_col, window, numrical_columns):
        df = self._filter(df, filter_col, window)
        df = self.aggregator.process(df, numrical_columns, self.user_id)
        return df


class SequenceGenerator:
    """
    Generate a sequence from transaction logs
    """

    def __init__(self, key_column, value_column):
        self.key_col = key_column
        self.val_col = value_column

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        d = {}
        for id_, df_ in df.groupby(self.key_col):
            d[id_] = " ".join(np.array(df_[self.val_col].values.tolist(), dtype=str))
        df["sequence"] = df[self.key_col].map(d)
        return df


class SequenceTransformer(BaseFitTransformer):
    """
    Transform a sequence into a vector processed with dimension reduction after TfIdf vectorized
    """

    def __init__(self, key_column):
        super().__init__()
        self.key_col = key_column

    def _tfidf(self, columns: list) -> None:
        self.tfv = TfidfVectorizer(
            max_features=100000, token_pattern="\w+", ngram_range=(1, 2)
        )

    def _svd(self) -> None:
        self.svd = TruncatedSVD(n_components=32, random_state=42)

    def fit_(self, df: pd.DataFrame, columns: list, target: str) -> None:
        df_user = df.loc[:, [self.key_col, "sequence"]].drop_duplicates()
        self._tfidf(columns)
        self._svd()
        df_ = self.tfv.fit_transform(df_user.loc[:, columns])
        self.svd.fit(df_)

    def transform_(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        df_user = df.loc[:, [self.key_col, "sequence"]].drop_duplicates()
        df_ = self.tfv.transform(df_user.loc[:, columns])
        df_ = self.svd.transform(df_)
        df_ = pd.DataFrame(df_).add_prefix(f"tfidf_svd_")
        df_.index = df_user[self.key_col]
        return df_


class Preprocessor:
    def __init__(self):
        self.te = None  # TargetEncoder
        self.st = None  # SequenceTransformer
        self.num_cols = NUMERICAL_COLUMNS
        self.cat_cols = CATEGORICAL_COLUMNS
        self.target_col = TARGET_COLUMN
        self.key_col = KEY_COLUMN
        self.val_col = VALUE_COLUMN
        self.ts_col = TIMESTAMP_COLUMN

    def preprocess(self, df, is_train):
        if is_train:
            self.te = TargetEncoder()
            self.st = SequenceTransformer(self.key_col)
        df = self.te.process(df, self.cat_cols, is_train, self.target_col, drop=True)
        df = TimestampAggregator().preprocess(
            df, self.ts_col, latest_column="latest", key_column=self.key_col
        )
        agt = LastNDayAggregator(self.key_col, self.ts_col, latest_column="latest")
        df = agt.preprocess(
            df, filter_col="ts_diff_days", window=1, numrical_columns=self.num_cols
        )
        df = SequenceGenerator(self.key_col, self.val_col).process(df)
        df = self.st.process(df, "sequence", is_train)
        print(df.head())
        return df
