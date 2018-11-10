import numpy as np
import time
from Analysis import Analysis
from Setup import Setup
from Xdb import Xdb
from Decorators import Decorators
from misc.xsave import mksavdir, save_result, make_filename
from helper import *


class Xana(Xdb, Setup, Analysis):

    def __init__(self, sample='', savdir='./', **kwargs):

        self.savdir = savdir
        self.sample = sample
        
        Xdb.__init__(self, **kwargs)
        Setup.__init__(self, **kwargs)
        Analysis.__init__(self, **kwargs)

    def mksavdir(self, *args, **kwargs):
        self.savdir = mksavdir(*args, **kwargs)
        self.dbfile = self.savdir+'Analysis_db.pkl'
        self.load_db(handle_existing='overwrite')


        
