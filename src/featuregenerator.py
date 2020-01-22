# import pandas as pd

# from abstractFeaturegenerator import (
#     Feature,
#     get_arguments,
#     generate_features,
# )
# from utils import load_yaml


# config = load_yaml("../config.yml")
# Feature.dir = config["path"]["feature"]


# class FeatureGenerator(Feature):
#     def create_features(self):
#         self.train["hoge"] = "hoge"
#         self.test["hoge"] = "hoge"


# if __name__ == "__main__":
#     args = get_arguments()

#     train = pd.read_csv("../input/nikkei/train.csv")
#     test = pd.read_csv("../input/nikkei/test.csv")

#     generate_features(globals(), args.overwrite)
