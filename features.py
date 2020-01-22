from src.abstractFeaturegenerator import (
    Feature,
    create_memo,
    get_arguments,
    generate_features,
)
from src.utils import mkdir, load_yaml


config = load_yaml()
Feature.dir = config["path"]["feature"]


class FeatureGenerator(Feature):
    def create_features(self):
        self.columns = ["user_id"]
        self.load(self.columns)
        self.train["hoge"] = "hoge"
        self.test["hoge"] = "hoge"
        create_memo("hoge", "hogehoge")


if __name__ == "__main__":
    # CSVのヘッダーを書き込み
    create_memo("特徴量", "メモ")

    mkdir(Feature.dir)
    args = get_arguments()

    generate_features(globals(), args.overwrite)
