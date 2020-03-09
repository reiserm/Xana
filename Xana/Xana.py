import os
import pickle
from .Analysis import Analysis
from .Setup import Setup
from .Xdb import Xdb
from .misc.xsave import mksavdir, save_result, make_filename


class Xana(Xdb, Analysis):
    """Xana class to perform XPCS, XSVS and SAXS data analysis.
    """

    def __init__(self, sample='', savdir='./', setupfile=None, **kwargs):
        """__init__ of Xana.

        Args:
            sample (str): sample name (optional).
            **kwargs: kwargs passed to initialize the database (Xdb), the
                setup (Setup) and the Analysis class (Analysis).
        """

        self.savdir = savdir #: str: directory to save results in.
        self.sample = sample #: str: sample name appearing in the database.
        self.setupfile = setupfile
        """str: setupfile to load at the beginning, can also be loaded with loadsetup method."""

        Xdb.__init__(self, **kwargs)
        if self.setupfile is None:
            self.setup = Setup(**kwargs)
        else:
            self.loadsetup(self.setupfile)
        Analysis.__init__(self, **kwargs)

    def mksavdir(self, savdir, *args, **kwargs):
        """Create directory for saving results.

        Args:
            savdir (str): name of the directory for saving the results.
            **kwargs: kwargs can be `savhome` and `handle_existing`. The former is the
                directory in which savdir will be created. The latter sets the default
                behavior to use existing directories, raise and error or overwrite
                files.
        """

        self.savdir = mksavdir(savdir, *args, **kwargs)
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
