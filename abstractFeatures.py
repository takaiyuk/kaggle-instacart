from src.abstractFeaturegenerator import (
    Feature,
    create_memo,
    get_arguments,
    generate_features,
    save_column,
)
from src.utils import mkdir, load_yaml


config = load_yaml()
Feature.dir = config["path"]["feature"]

"""
class FeatureGenerator(Feature):
    def create_features(self):
        # self.columns には特徴量生成に必要な列名を書く
        self.columns = ["user_id"]
        self.load(self.columns)
        self.train["hoge"] = "hoge"
        self.test["hoge"] = "hoge"
        create_memo("feature name", "feature name memo")
"""

if __name__ == "__main__":
    # train / test を列ごとに分割して保存しておく
    save_column()

    # CSVのヘッダーを書き込み
    create_memo("feature", "memo")

    mkdir(Feature.dir)
    args = get_arguments()

    generate_features(globals(), args.overwrite)
