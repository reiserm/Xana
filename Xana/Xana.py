import os
import numpy as np
import time
import pickle
from .Analysis import Analysis
from .Setup import Setup
from .Xdb import Xdb
from .Decorators import Decorators
from .misc.xsave import mksavdir, save_result, make_filename
from .helper import *



class Xana(Xdb, Analysis):

    def __init__(self, sample='', savdir='./', setupfile=None, **kwargs):

        self.savdir = savdir
        self.sample = sample
        self.setupfile = setupfile
        Xdb.__init__(self, **kwargs)
        if self.setupfile is None:
            self.setup = Setup(**kwargs)
        else:
            self.loadsetup(self.setupfile)
        Analysis.__init__(self, **kwargs)

    def mksavdir(self, *args, **kwargs):
        self.savdir = mksavdir(*args, **kwargs)
        self.setup.savdir = self.savdir
        self.dbfile = self.savdir+'Analysis_db.pkl'
        self.load_db(handle_existing='overwrite')

    def loadsetup(self, filename=None):
        if filename is None:
            print('No setup defined.')
        else:
            path, fname = make_filename(self, 'setupfile', filename)
            fname = path + fname
            if os.path.isfile(fname):
                self.setupfile = fname
                self.setup = pickle.load(open(self.setupfile, 'rb'))
                print('Loaded setupfile:\n\t{}.'.format(self.setupfile))
            else:
                self.setupfile = None
                print('No setup defined.')

    def savesetup(self, filename=None, **kwargs):
        if filename is None:
            filename = input('Enter filename for setupfile.\t')
#        savd = copy.deepcopy(vars(self))
        self.setupfile = save_result(self.setup, 'setup', self.savdir, filename, **kwargs)
        


        
