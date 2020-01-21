import argparse
import gc
import joblib
import os
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
    def __init__(self, parser, config):
        self.parser = parser
        self.config = config
        self.debug = parser["debug"]
        self.model = parser["model"]

        self.logger, self.version = Logger(self.config["path"]["logs"]).get_logger()
        self.timer = Timer()
        self.dataloader = DataLoader(self.config)
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
                train = self.preprocessor.preprocess(train, is_train=True)
                joblib.dump(train, "./input/preprocess/train.jbl", compress=3)
                del train
                gc.collect()
                test = self.preprocessor.preprocess(test, is_train=True)
                joblib.dump(test, "./input/preprocess/test.jbl", compress=3)
                del test
                gc.collect()

            with self.timer.timer("Process Estimator"):
                train = self.dataloader.load_joblib("./input/preprocess/train.jbl")
                self.estimator.estimate(train, is_train=True)
                del train
                gc.collect()

            with self.timer.timer("Process Estimator - prediction"):
                test = self.dataloader.load_joblib("./input/preprocess/test.jbl")
                self.estimator.estimate(test, is_train=False)
                del test
                gc.collect()

            with self.utils.timer("Process Submission"):
                submit = self.loader.load_sample_submission()
                submit[self.config["column"]["target"]] = self.estimator.pred_test
                submit.to_csv(
                    f'{self.config["path"]["submit"]}/submission_{self.version}.csv'
                )

        except Exception as e:
            t, v, tb = sys.exc_info()
            x = traceback.format_exception(t, v, tb)
            self.logger.info(f"error occurred: {''.join(x)}")
            raise e


if __name__ == "__main__":
    config = load_yaml()
    for k, v in config["path"].items():
        # path が file ではなく directory の場合 mkdir する
        if os.path.splitext(os.path.basename(v))[1] == "":
            mkdir(v)
    Runner(p, config).run()
