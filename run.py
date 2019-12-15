import argparse
import gc
import pandas as pd
import traceback
import sys
import warnings

pd.set_option("display.max_columns", 50)
warnings.filterwarnings("ignore")

from src.utils import Logger, Utils
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.estimator import LightgbmParameter, Estimator
from src.const import TARGET_COLUMN, UNUSED_COLUMNS, SUBMIT_PREFIX

parser = argparse.ArgumentParser(description="argparse for run.py")
parser.add_argument("--debug", action="store_true", help="debug mode")
p = vars(parser.parse_args())


class Runner:
    def __init__(self, parser):
        self.debug_mode = parser["debug"]
        self.logger, self.version = Logger().get_logger()
        self.utils = Utils()
        self.dataloader = DataLoader()
        self.preprocessor = Preprocessor()
        self.parameter = LightgbmParameter(self.debug_mode)
        self.estimator = Estimator(self.parameter, self.logger, self.version)

        self.logger.info(f'debug mode: {parser["debug"]}')
        self.nrows = None
        if self.debug_mode:
            self.nrows = 100000

    def run(self):
        try:
            with self.utils.timer("Process loading train"):
                train = self.dataloader.load_train(nrows=self.nrows)
            with self.utils.timer("Process preprocessing train"):
                train = self.preprocessor.preprocess(train, is_train=True)
            with self.utils.timer("Process fitting train"):
                self.estimator.kfold_fit(
                    train.drop(["user_id"] + [TARGET_COLUMN], axis=1),
                    train[TARGET_COLUMN],
                )
            self.estimator.plot_feature_importance()
            del train
            gc.collect()

            with self.utils.timer("Process loading test"):
                test = self.dataloader.load_test()
                test[TARGET_COLUMN] = 0
            with self.utils.timer("Process preprocessing test"):
                test = self.preprocessor.preprocess(test, is_train=False)
            with self.utils.timer("Process predicting test"):
                X_test = test.drop(["user_id"] + [TARGET_COLUMN], axis=1)
                self.estimator.kfold_predict(X_test=X_test)
                del X_test
                gc.collect()
                test["pred"] = self.estimator.pred_test
                test_agg = (
                    test.loc[:, ["user_id", "pred"]].groupby("user_id").pred.mean()
                )
            print(test_agg.head())
            assert set(test_agg.index.values) == set(
                self.dataloader.load_sample_submission().index.values
            )

            with self.utils.timer("Process generating submission"):
                test_agg.to_csv(f"{SUBMIT_PREFIX}/submission_{self.version}.csv")

        except Exception as e:
            t, v, tb = sys.exc_info()
            x = traceback.format_exception(t, v, tb)
            self.logger.info(f"error occurred: {''.join(x)}")
            raise e


if __name__ == "__main__":
    Runner(p).run()
