from .abstractFeaturegenerator import (
    Feature,
    get_arguments,
    get_features,
    generate_features,
)


class FeatureGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        pass
