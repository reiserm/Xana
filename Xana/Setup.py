from misc.xsave import save_result, make_filename
import pickle
import os
import numpy as np

class Setup:

    def __init__(self, setupfile, savdir='./', **kwargs):

        self.savdir = self.__dict__.get('savdir', savdir)
        self.setupfile = setupfile
        self.setup = None
        self.load_setup()

    def load_setup(self, filename=None):
        path, fname = make_filename(self, 'setupfile', filename)
        fname = path + fname
        if os.path.isfile(fname):
            self.setupfile = fname
            self.setup = pickle.load(open(self.setupfile, 'rb'))
            print('Loaded setupfile:\n\t{}.'.format(self.setupfile))
        else:
            self.setupfile = None
            print('No setup defined.')

    def make_setup(self, **kwargs):
        keys = ['ctr', 'distance', 'lambda', 'pix_size']
        mesg = [('beam center x [pixel]', 'beam center y [pixel]'),
                ('sample-detector distance [m]',),
                ('wavelength [A]',),
                ('pixel size x [um]', 'pixel size y [um]')
                ]
        setup = {}
        for k, m in zip(keys, mesg):
            if k in kwargs:
                inp = kwargs[k]
            else:
                inp = []
                for mi in m:
                    inp.append(input(mi+'\t'))
            if k == 'ctr':
                inp = np.array(inp).astype('int32')
            else:
                inp = np.array(inp).astype('float32')
            setup[k] = inp
        self.setup = setup

    def save_setup(self, filename=None, **kwargs):
        if filename is None:
            filename = input('Enter filename for setupfile.\t')
        self.setupfile = save_result(
            self.setup, 'setup', self.savdir, filename, **kwargs)
