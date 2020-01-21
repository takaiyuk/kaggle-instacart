from abstractPreprocessor import (
    AbstractPreprocessor,
    LastNDayAggregator,
    SequenceGenerator,
    SequenceTransformer,
    TargetEncoder,
    TimestampAggregator,
)


class Preprocessor(AbstractPreprocessor):
    def __init__(self, config):
        super().__init__(self, config)

    def preprocess(self, , df, is_train):
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
