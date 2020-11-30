import pyFAI


class Eiger500k(pyFAI.detectors.Eiger):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.aliases = ["Eiger500k"]
        self.dim = self.shape = (514, 1030)
