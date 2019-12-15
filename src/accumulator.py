import category_encoders as ce
import pandas as pd

from .const import (
    NUMERICAL_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET_COLUMN,
    KEY_COLUMN,
    TIMESTAMP_COLUMN,
)


class Accumulator:
    def __init__(self):
        self.num_cols = NUMERICAL_COLUMNS
        self.cat_cols = CATEGORICAL_COLUMNS
        self.target_col = TARGET_COLUMN
        self.key_col = KEY_COLUMN
        self.ts_col = TIMESTAMP_COLUMN

    def Accumulate(self, df):
        df_user = (
            df.loc[:, [self.key_column, self.target_col]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        df_user
