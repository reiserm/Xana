import numpy as np
import pyFAI


class Agipd1m(pyFAI.detectors.Detector):
    def __init__(self, *args, **kwargs):

        super().__init__(
            pixel1=2e-4,
            pixel2=2e-4,
            max_shape=(8192, 128),
        )
        self.shape = self.dim = (8192, 128)
        self.aliases = ["Agipd1m"]
        self.IS_CONTIGUOUS = False
        self.mask = np.zeros(self.shape)

        # self._pixel_corners = spatial_distortion.astype(np.float32)
