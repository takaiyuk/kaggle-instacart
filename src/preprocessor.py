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
        df = self.st.process(df, ["sequence"], is_train)
        print(df.head())
        return df
