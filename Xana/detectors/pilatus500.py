import pyFAI


class Pilatus500(pyFAI.detectors.Pilatus500):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.aliases = ["Pilatus500"]
        self.dim = self.shape = (619, 487)
