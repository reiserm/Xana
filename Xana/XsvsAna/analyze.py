import numpy as np
import multiprocessing as mp
from queue import PriorityQueue
from XsvsAna.pyxsvs3 import pyxsvs
from misc.xsave import save_result
from multiprocessing.managers import SyncManager
from helper import *


class MyManager(SyncManager):
    pass


def Manager():
    m = MyManager()
    m.start()
    return m


def xsvs(
    obj,
    series_id,
    first=0,
    last=None,
    filename="",
    method="full",
    nbins=15,
    handle_existing="next",
    nprocs=4,
    nread_procs=4,
    chunk_size=100,
    dark=None,
    load_verbose=False,
    dtype=np.float32,
    **kwargs
):

    # MyManager.register("PriorityQueue", PriorityQueue)  # Register a shared PriorityQueue

    for sid in series_id:
        print(
            "\n\n#### Starting XSVS Analysis ####\nSeries: {} in folder {}\n".format(
                sid, obj.xdata.datdir
            )
        )

        if last is None:
            last = obj.xdata.meta.loc[sid, "nframes"]

        if dark is not None:
            if type(dark) == int:
                print("Loading DB entry {} as dark.".format(dark))
                dark = obj.get_item(dark)["Isaxs"]

        read_data_opt = {
            "first": (first,),
            "last": (last,),
            "dark": dark,
            "verbose": load_verbose,
            "dtype": dtype,
            "qsec": obj.setup["qsec"],
            "output": "2dsection",
            "nprocs": 1,
        }
        read_data_opt.update(kwargs)

        qsec = obj.setup["qsec"]
        xpcs_opt = {
            "nimages": last - first,
            "dim": (qsec[1][0] - qsec[0][0] + 1, qsec[1][1] - qsec[0][1] + 1),
        }

        t_e = 0
        for attr in ["t_exposure", "pulseLength"]:
            if attr in obj.xdata.meta.columns:
                item = obj.xdata.meta.loc[sid, attr]
                if attr == "pulseLength":
                    t_e += item * 1e15
                else:
                    t_e += item

        fmax = obj.xdata.meta.loc[sid, "nframes"]
        chunks = [
            np.arange(
                first + i * chunk_size,
                min([min(first + (i + 1) * chunk_size, last), fmax]),
            )
            for i in range(np.ceil((last - first) / chunk_size).astype(np.int32))
        ]

        print("Using {} processes to read data.".format(nread_procs))
        #        m = Manager()
        # dataQ = m.PriorityQueue(nread_procs)
        # indxQ = m.PriorityQueue()
        dataQ = mp.Queue(nread_procs)
        indxQ = mp.Queue()
        read_data_opt["method"] = "queue_chunk"
        read_data_opt["dataQ"] = dataQ
        read_data_opt["indxQ"] = indxQ
        xpcs_opt["dataQ"] = dataQ

        for i, chunk in enumerate(chunks):
            indxQ.put((i, chunk))

        lock = 0
        if "h5" in obj.xdata.fmtstr:
            lock = mp.Lock()
            read_data_opt["lock"] = lock

        procs = []
        for ip in range(nread_procs):
            procs.append(
                mp.Process(
                    target=obj.xdata.get_series, args=(sid,), kwargs=read_data_opt
                )
            )
            procs[ip].start()

        probd = pyxsvs(
            xpcs_opt,
            obj.setup["qroi"],
            method=method,
            nbins=nbins,
            t_e=t_e,
            qv=obj.setup["qv"],
            nprocs=nprocs,
            qsec=obj.setup["qsec"][0],
        )

        # stopping processes
        for ip in range(nread_procs):
            procs[ip].join()

        # closing queues
        dataQ.close()
        dataQ.join_thread()
        indxQ.close()
        indxQ.join_thread()

        print(obj.xdata.datdir)
        f = (
            # obj.xdata.datdir.split("/")[-2]
            "s"
            + str(obj.xdata.meta.loc[sid, "series"])
            + filename
        )
        savfile = save_result(
            probd, "xsvs", obj.savdir, f, handle_existing=handle_existing
        )
        obj.add_db_entry(sid, savfile)
