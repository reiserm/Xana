import os
import pickle
from .Analysis import Analysis
from .Setup import Setup
from .Xdb import Xdb
from .misc.xsave import mksavdir, save_result, make_filename


class Xana(Xdb, Analysis):
    """Xana class to perform XPCS, XSVS and SAXS data analysis.

    Args:
        sample (str, optional): sample name.
        savdir (str, optional): directory to save results.
        setupfile (str, optional): setupfile to load.
        maskfile (str, optional): maskfile to load in `'.npy'` format.
        detector (str, optional): detector used for the measurement.
        dbfile (str, optional): database file.
        fmtstr (str, optional): Specify the format to load data.
    """

    def __init__(self, sample='', savdir='./', setupfile=None, maskfile=None,
                 detector='eiger500k', dbfile=None, datdir=None, fmtstr=None):

        self.savdir = savdir
        self.sample = sample
        self.setupfile = setupfile

        Xdb.__init__(self, dbfile=dbfile)
        if self.setupfile is None:
            self.setup = Setup(maskfile=maskfile, detector=detector)
        else:
            self.loadsetup(self.setupfile)
        Analysis.__init__(self, datdir=datdir, fmtstr=fmtstr)

    def __str__(self):
        return ('Xana Instance\n' +
                "savdir: {}\n" +
                "sample name: {}\n" +
                "database file: {}\n" +
                'setup file: {}\n').format(self.savdir, self.sample,
                                           self.dbfile, self.setupfile)

    def __repr__(self):
        return self.__str__()

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
