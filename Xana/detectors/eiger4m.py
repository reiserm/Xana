import pyFAI


class Eiger4m(pyFAI.detectors.Eiger):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.aliases = ["Eiger4M"]
        self.dim = self.shape = (2167, 2070)
