import argparse
import gc
import pandas as pd
import traceback
import sys
import warnings

pd.set_option("display.max_columns", 100)
warnings.filterwarnings("ignore")

from src.utils import Logger, Timer, load_yaml, mkdir
from src.dataloader import DataLoader
from src.preprocessor import Preprocessor
from src.estimator import Estimator

parser = argparse.ArgumentParser(description="argparse for run.py")
parser.add_argument("--debug", action="store_true", help="debug mode")
parser.add_argument(
    "--model",
    default="lgb",
    required=False,
    choices=["lgb", "cb", "xgb", "nn", "linear"],
    help="model type",
)
p = vars(parser.parse_args())


class Runner:
    def __init__(self, parser):
        self.parser = parser
        self.config = load_yaml()
        self.debug = parser["debug"]
        self.model = parser["model"]

        self.logger, self.version = Logger().get_logger()
        self.timer = Timer()
        self.dataloader = DataLoader()
        self.preprocessor = Preprocessor()
        self.estimator = Estimator(self.config, self.logger, self.model, self.version)

        self.nrows = None
        if self.debug:
            self.nrows = 100000

        for k, v in self.parser.items():
            self.logger.info(f"{k}: {v}")

    def run(self):
        try:
            with self.timer.timer("Process Loader"):
                train = self.dataloader.load_train(nrows=self.nrows)
                test = self.dataloader.load_test()

            with self.timer.timer("Process Preprocessor"):
                self.preprocessor.preprocess(train, test, is_train=True)

            with self.timer.timer("Process Estimator - training"):
                train = self.dataloader.load_joblib("train.jbl")
                self.estimator.kfold_fit(
                    train.drop(self.config["column"]["target"], axis=1),
                    train[self.config["column"]["target"]],
                )
                self.estimator.plot_feature_importance()
                del train
                gc.collect()

            with self.timer.timer("Process Estimator - prediction"):
                X_test = self.dataloader.load_joblib("test.jbl")
                self.estimator.kfold_predict(X_test=X_test)
                del X_test
                gc.collect()
                pred_test = self.estimator.pred_test

            with self.utils.timer("Process Submission"):
                submit = self.loader.load_sample_submission()
                submit[self.config["column"]["target"]] = pred_test
                submit.to_csv(
                    f'{self.config["path"]["submit"]}/submission_{self.version}.csv'
                )

        except Exception as e:
            t, v, tb = sys.exc_info()
            x = traceback.format_exception(t, v, tb)
            self.logger.info(f"error occurred: {''.join(x)}")
            raise e


if __name__ == "__main__":
    Runner(p).run()
