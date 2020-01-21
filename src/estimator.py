from .abstractEstimator import (
    AbstractCatboostEstimator,
    AbstractLightgbmEstimator,
    AbstractLinearEstimator,
    AbstractNeuralnetEstimator,
    AbstractXgboostEstimator,
)


class CatboostEstimator(AbstractCatboostEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass


class LightgbmEstimator(AbstractLightgbmEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass


class LinearEstimator(AbstractLinearEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass


class NeuralnetEstimator(AbstractNeuralnetEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass


class XgboostEstimator(AbstractXgboostEstimator):
    def __init__(self, config, logger, model_type, version):
        super().__init__(config, logger, model_type, version)

    # If you need to fix, implementet it.
    # def estimate(self, df: pd.DataFrame, is_train: bool) -> None:
    #     pass
