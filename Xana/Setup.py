from .misc.xsave import save_result, make_filename
from . import detectors
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
import pickle
import os
import numpy as np
import copy


class Setup:

    def __init__(self, **kwargs):
        
        self.detector = kwargs.pop('detector', None)
        self.maskfile = kwargs.pop('maskfile', None)
        self.mask = self.load_mask()
        self.wavelength = None
        self.distance = None
        self.center = None
        self.qv_init = None
        self.phiv_init = None
        self.dqv = None
        self.qroi = None
        self.gproi = None
        self.qsec = None
        self.qsec_center = None
        self.qsec_mask = None
        self.qsec_dim = None
        self.qsec_ai = None
        self.qv = None
        self.phiv = None
        self.radii = None

    @property
    def detector(self):
        return self.__detector

    @detector.setter
    def detector(self, name):
        if name is None:
            self.__detector = None
            print('No sepcific detector defined.')
        else:
            self.__detector = detectors.grab(name)
    
    def __getstate__(self):
        d = dict(vars(self))
        d['detector'] =  self.detector.aliases[0].lower()
        del d['ai'], d['qsec_ai'], d['_Setup__detector']
        return d

    def __setstate__(self, d):
        if 'lambda' in d: # handle old syntax
            d['wavelengt'] = d.pop('lambda')
        self.detector = d.pop('detector', None)
        self.__dict__.update(d)
        self.ai = self.update_ai()
        if d['qsec'] is not None:
            self.qsec_ai = self.update_ai(self.qsec_center)
        else:
            self.qsec_ai = None

    def make(self, **kwargs):
        keys = ['center', 'distance', 'wavelength',]
        mesg = [('beam center x [pixel]', 'beam center y [pixel]'),
                ('sample-detector distance [m]',),
                ('wavelength [A]',),
                ]
        setup = {}
        for k, m in zip(keys, mesg):
            if k in kwargs:
                inp = kwargs[k]
            else:
                inp = []
                for mi in m:
                    inp.append(input(mi+'\t'))
            if k == 'center':
                inp = np.array(inp).astype('int32')
            else:
                inp = np.array(inp).astype('float32')
            setup[k] = inp
        self.__dict__.update(setup)
        self.ai = self.update_ai()

    def update_ai(self, center=None, nbins=1000):
        
        if center is None:
            center = self.center

        ai = AzimuthalIntegrator(detector=self.detector, dist=self.distance)
        ai.setFit2D(self.distance*1000, center[0], center[1])
        ai.wavelength = self.wavelength * 1e-10
        return ai

    def load_mask(self):
        self.maskfile = os.path.abspath(self.maskfile)
        if self.maskfile.endswith('edf'):
            mask = loadedf(self.maskfile)
        elif self.maskfile.endswith('npy'):
            mask = np.load(self.maskfile)
        else:
            print('Mask file not found. Continuing without mask.')
            mask = None
        return mask
