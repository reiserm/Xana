import numpy as np
import time
import copy
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from queue import PriorityQueue
from .XpcsAna.Xpcs import Xpcs
from .XsvsAna.Xsvs import Xsvs
from .SaxsAna.Saxs import Saxs
from .ProcData.Xdata import Xdata
from .Decorators import Decorators
from .misc.xsave import save_result


class MyManager(SyncManager):
    pass


def Manager():
    m = MyManager()
    m.start()
    return m


class Analysis(Xdata):
    """Main class to compute the data analysis."""

    def __init__(self, datdir=None, fmtstr=None):
        super().__init__(datdir=fmtstr, fmtstr=fmtstr)

    @Decorators.input2list
    def analyze(
        self,
        series_id,
        method,
        first=0,
        last=np.inf,
        handle_existing="next",
        nread_procs=1,
        chunk_size=200,
        verbose=True,
        dark=None,
        dtype=np.float32,
        filename="",
        read_kwargs={},
        **kwargs,
    ):
        """Perform the analysis.

        Args:
            series_id (int): Index of the dataset in the :code:`Xana.meta` table.
            method (str): Which analysis should be performed. Currently the options are:

                +----------+-------------------------+
                | method   |    analysis             |
                +==========+=========================+
                | saxs     | azimuthal intensity     |
                +----------+-------------------------+
                | xpcs     | correlation functions   |
                +----------+-------------------------+
                | xpcs_evt | event correlator        |
                +----------+-------------------------+
                | xsvs     | photon probabilities    |
                +----------+-------------------------+

            first (int, optional): Index of the first image to analyze. Defaults 0.
            last (int, optional): Index of the last image to analyze. Defaults :code:`nf-1`
                where :code:`nf` is the number of images of the series.
            handle_existing (str, optional): how to treat existing files. Defaults to :code:`next`
                meaning that for each saved file a new filename is created with a counter.
            nread_procs (int, optional): How many processes are spawned to read the data.
                Defaults to 1.
            chunk_size (int, optional): Load the data in chunks of this many images.
            verbose (bool, optional): Print more detailed output if True (default).
            read_kwargs (dict, optional): Additional kwargs passed to the data reader.
            **kwargs: Additional kwargs are passed to the particular analysis routine depending
                on the value of :code:`method`.

        """

        if not self.setup.wavelength:
            raise ValueError("Setup is not defined properly. Cannot perform analysis.")

        for sid in series_id:
            if verbose:
                print(
                    "\n\n#### Starting %s Analysis ####\nSeries: %d in folder %s\n"
                    % (method, sid, self.datdir)
                )
                print("Using {} processes to read data.".format(nread_procs))

            # copy the metadata
            self._meta_save = copy.deepcopy(self.meta)
            rois = copy.deepcopy(self.setup.qroi)

            # if dark is not None:
            #     if type(dark) == int:
            #         print('Loading DB entry {} as dark.'.format(dark))
            #         dark = self.xana.get_item(dark)['Isaxs']

            nf = self.meta.loc[sid, "nframes"]
            first_proc = first % nf + self.meta.loc[sid, "first"]
            last_proc = min([self.meta.loc[sid, "nframes"], last])
            last_proc = (last_proc - 1) % nf + self.meta.loc[sid, "first"]
            self._meta_save.loc[sid, ["first", "last", "nframes"]] = (
                first_proc,
                last_proc,
                last_proc - first_proc + 1,
            )

            # update meta database
            # self.meta.loc[sid, 'first'] = first
            # self.meta.loc[sid, 'last'] = last

            # dict with options and variables passed to the data reader
            read_opt = {
                "first": first_proc,
                "last": last_proc,
                "dark": dark,
                "verbose": False,
                "dtype": dtype,
                "qsec": self.setup.qsec,
                "output": "2dsection",
                "nprocs": nread_procs,
                "chunk_size": chunk_size,
            }
            saxs_dict = read_opt.copy()
            read_opt.update(read_kwargs)

            proc_dat = {
                "nimages": self._meta_save.loc[sid, "nframes"],
                "dim": self.setup.qsec_dim,
            }

            # old chunks
            # chunks = [np.arange(first + i*chunk_size, min(first + (i + 1)*chunk_size, last))
            #           for i in range(np.ceil((last - first) / chunk_size).astype(np.int32))]

            # new chunks
            ind_arange = np.arange(first_proc, last_proc + 1)
            bins = np.arange(0, nf, chunk_size)
            digitized = np.digitize(ind_arange, bins)
            chunks = [
                ind_arange[np.where(digitized == i)] for i in np.unique(digitized)
            ]

            if method in ["xpcs", "xsvs"]:

                # Register a shared PriorityQueue
                MyManager.register("PriorityQueue", PriorityQueue)
                m = Manager()
                dataQ = m.PriorityQueue(nread_procs)
                indxQ = m.PriorityQueue()
                # dataQ = mp.Queue(nread_procs)
                # indxQ = mp.Queue()'symmetric_whole'

                # add queues to read and process dictionaries
                read_opt["dataQ"] = dataQ
                read_opt["indxQ"] = indxQ
                read_opt["method"] = "queue_chunk"
                proc_dat["dataQ"] = dataQ

                for i, chunk in enumerate(chunks):
                    indxQ.put((i, chunk))

                # h5 files can only be opened by one process at a time and, therefore,
                # the processes have to acquire a lock for reading data
                lock = 0
                if "h5" in self.fmtstr:
                    lock = mp.Lock()
                    read_opt["lock"] = lock

                procs = []
                for ip in range(nread_procs):
                    procs.append(
                        mp.Process(target=self.get_series, args=(sid,), kwargs=read_opt)
                    )
                    procs[ip].start()
                    time.sleep(2)

            if method == "xpcs":
                saxs = kwargs.pop("saxs", "compute")
                Isaxs = self._get_xpcs_args(sid, saxs, saxs_dict)
                dt = self._get_delay_time(sid)

                nprocs = max([2, kwargs.pop("nprocs", 2)])
                savd = Xpcs.pyxpcs(
                    proc_dat,
                    rois,
                    dt=dt,
                    qv=self.setup.qv,
                    saxs=Isaxs,
                    mask=self.setup.mask,
                    ctr=self.setup.center,
                    qsec=self.setup.qsec[0],
                    nprocs=nprocs,
                    **kwargs,
                )

            elif method == "xpcs_evt":
                dt = self._get_delay_time(sid)
                evt_dict = dict(
                    method="events",
                    verbose=True,
                    qroi=rois,
                    dtype=np.uint32,
                )
                read_opt.update(evt_dict)
                evt = self.get_series(sid, **read_opt)
                savd = Xpcs.eventcorrelator(
                    evt[1:], rois, self.setup.qv, dt, method="events", **kwargs
                )

            elif method == "xsvs":

                t_e = self._get_xsvs_args(
                    sid,
                )
                savd = Xsvs.pyxsvs(
                    proc_dat,
                    rois,
                    t_e=t_e,
                    qv=self.setup.qv,
                    qsec=self.setup.qsec[0],
                    **kwargs,
                )

            elif method == "saxs":

                read_opt["output"] = "2d"
                proc_dat = {
                    "get_series": self.get_series,
                    "sid": sid,
                    "setup": self.setup,
                    "mask": self.setup.mask,
                }
                savd = Saxs.pysaxs(proc_dat, **read_opt, **kwargs)

            else:
                raise ValueError("Analysis type %s not understood." % method)

            if method in ["xpcs", "xsvs"]:
                # stopping processes
                for ip in range(nread_procs):
                    procs[ip].join()

                # closing queues
                # dataQ.close()
                # dataQ.join_thread()
                # indxQ.close()
                # indxQ.join_thread()

            f = f's{self.meta.loc[sid, "series"]:04}{filename}'
            savfile = save_result(savd, method, self.savdir, f, handle_existing)

            if self.db is None:
                self.init_db()
            self.add_db_entry(sid, savfile, method)

    def _get_xpcs_args(self, sid, saxs, read_opt):
        """Get Saxs and delay time for XPCS analysis."""
        if saxs == "compute":
            print("Calculating average SAXS image.")
            Isaxs = self.get_series(sid, method="average", **read_opt)[0]
        elif isinstance(saxs, int):
            if saxs == -1:
                saxs = self.db.shape[0] - 1
            print("Loading average SAXS from database entry {}".format(saxs))
            Isaxs = self.get_item(saxs)["Isaxs"]
        else:
            Isaxs = saxs
        return Isaxs

    def _get_delay_time(self, sid):
        dt = 0
        for attr in [
            "t_delay",
            "t_exposure",
            "t_readout",
            "t_latency",
            "rate",
            "pulseLength",
        ]:
            if attr in self.meta.columns:
                item = self.meta.loc[sid, attr]
                if attr == "rate":
                    dt += 1 / item
                elif attr == "pulseLength":
                    dt += item * 1e-15
                else:
                    dt += item
                    if attr == "t_delay":
                        break
        return dt

    def _get_xsvs_args(self, sid):
        """Get exposure time for XSVS analysis"""
        t_e = 0
        for attr in ["t_exposure", "pulseLength"]:
            if attr in self.meta.columns:
                item = self.meta.loc[sid, attr]
                if attr == "pulseLength":
                    t_e += item * 1e15
                else:
                    t_e += item

        return t_e

    def defineqrois(self, input_, **kwargs):
        if type(input_) == int:
            Isaxs = self.get_item(input_)["Isaxs"]
        elif type(input_) == np.ndarray:
            Isaxs = input_
        if Isaxs.ndim == 3:
            Isaxs = self.arrange_tiles(Isaxs)
        Saxs.defineqrois(self.setup, Isaxs, **kwargs)

    @staticmethod
    def find_center(*args, **kwargs):
        return Saxs.find_center(*args, **kwargs)
