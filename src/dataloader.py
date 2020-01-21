from .abstractDataloader import AbstractDataLoader


class DataLoader(AbstractDataLoader):
    def __init__(self, config):
        super().__init__(self, config)

    def load(self):
        pass
