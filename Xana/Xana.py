import numpy as np
import time
from Analysis import Analysis
from Setup import Setup
from Xdb import Xdb
from Decorators import Decorators
from misc.xsave import mksavdir, save_result, make_filename
from helper import *


class Xana(Xdb, Setup, Analysis):

    def __init__(self, **kwargs):
        
        setupfile = kwargs.pop('setupfile', '')
        sample = kwargs.pop('sample', '')
        Xdb.__init__(self, sample, setupfile)
        Setup.__init__(self, setupfile, **kwargs)
        Analysis.__init__(self, **kwargs)

