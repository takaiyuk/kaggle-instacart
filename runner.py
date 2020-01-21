import argparse
import os

from abstractRunner import AbstractRunner
from src.utils import mkdir, load_yaml

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


class Runner(AbstractRunner):
    def __init__(self, parser, config):
        super().__init__(parser, config)


if __name__ == "__main__":
    config = load_yaml()
    for k, v in config["path"].items():
        # path が file ではなく directory の場合 mkdir する
        if os.path.splitext(os.path.basename(v))[1] == "":
            mkdir(v)
    Runner(p, config).run()
