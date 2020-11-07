import pyFAI


class Maxipix(pyFAI.detectors.Maxipix2x2):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.aliases = ["Maxipix"]
        self.dim = self.shape = (516, 516)
