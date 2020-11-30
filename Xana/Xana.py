import os
from pathlib import Path
import pickle
from .Analysis import Analysis
from .Setup import Setup
from .Xdb import Xdb
from .misc.xsave import mksavdir, save_result, make_filename


class Xana(Xdb, Analysis):
    """Xana class to perform XPCS, XSVS and SAXS data analysis.

    Note:
        Different kwargs are necessary for data loading, processing and interpretation.

        * Data loading: :code:`fmtstr` has to be provided.
        * Data processing: additionally, :code:`detector` and :code:`setupfile` have to be provided.
        * Data interpretation: :code:`setupfile` and :code:`dbfile` should be passed.

        In principle, the setupfile and the database file can also be loaded afterwards.

    Args:
        sample (str, optional): sample name that appears in the database.
        setupfile (str, optional): setupfile created previously by the
            :code:`Xana.savesetup` method storing the setup parameters.
        maskfile (str, optional): maskfile to load in :code:`.npy` format. Good pixels
            are set to 1, bad ones to 0.
        fmtstr (str, optional): Specify the format to load data. Valid options are:

            +-----------------------+----------+------------+
            | fmtstr                | beamline | facility   |
            +=======================+==========+============+
            |id10_eiger_single_edf  | ID10     | ESRF       |
            +-----------------------+----------+------------+
            |p10_eiger_h5           | P10      | PETRA-III  |
            +-----------------------+----------+------------+
            |lambda_nxs             | P10      | PETRA-III  |
            +-----------------------+----------+------------+
            |id02_eiger_single_edf  | ID02     | ESRF       |
            +-----------------------+----------+------------+
            |id02_eiger_multi_edf   | ID02     | ESRF       |
            +-----------------------+----------+------------+

        detector (str, optional): detector used for the measurement. Options are:
            :code:`'eiger500k'`, :code:`'eiger1m'`, :code:`'agipd1m'`.

        savdir (str, optional): directory to save results.
        dbfile (str, optional): database file.
    """

    def __init__(
        self,
        sample="",
        setupfile=None,
        maskfile=None,
        fmtstr=None,
        detector="eiger500k",
        savdir="./",
        dbfile=None,
        datdir=None,
    ):

        self.savdir = Path(savdir).resolve()
        self.sample = sample
        self.setupfile = setupfile

        Xdb.__init__(self, dbfile=dbfile)
        if self.setupfile is None:
            self.setup = Setup(maskfile=maskfile, detector=detector)
        else:
            self.loadsetup(self.setupfile)
        Analysis.__init__(self, datdir=datdir, fmtstr=fmtstr)

    def __str__(self):
        return (
            "Xana Instance\n"
            + "savdir: {}\n"
            + "sample name: {}\n"
            + "database file: {}\n"
            + "setup file: {}\n"
        ).format(self.savdir, self.sample, self.dbfile, self.setupfile)

    def __repr__(self):
        return self.__str__()

    def mksavdir(self, name, basepath="./"):
        """Create directory for saving output.

        Args:
            name (str): name of the directory for saving the results.
            basepath (str, optional): parent path to the directory. Defaults to `'./'`.
        """

        self.savdir = Path(mksavdir(name, basepath)).resolve()
        self.setup.savdir = self.savdir
        self.dbfile = Path(self.savdir).joinpath("analysis_database.pkl")
        self.load_db(handle_existing="overwrite")

    def loadsetup(self, filename=None):

        if filename is None:
            print("No setup defined.")
        else:
            fname = make_filename(self, "setupfile", filename)
            if fname.is_file():
                self.setupfile = fname
                self.setup = pickle.load(open(self.setupfile, "rb"))
                print("Loaded setupfile:\n\t{}.".format(self.setupfile))
            else:
                self.setupfile = None
                print("No setup defined.")

    def savesetup(self, filename=None, **kwargs):
        if filename is None:
            filename = input("Enter filename for setupfile.\t")
        #        savd = copy.deepcopy(vars(self))
        self.setupfile = save_result(
            self.setup, "setup", self.savdir, filename, **kwargs
        )
