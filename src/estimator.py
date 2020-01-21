from .abstractEstimator import LightgbmEstimator

DROP_COLUMNS = []


class Estimator(LightgbmEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass
