import numpy as np
import multiprocessing as mp
from queue import PriorityQueue
from multiprocessing.managers import SyncManager
from ..helper import *
import time
import h5py as h5
from ..misc.progressbar import progress


class MyManager(SyncManager):
    pass


def Manager():
    m = MyManager()
    m.start()
    return m


def init_h5_file():
    pass


def h5_dump(fname, nframes, dataQ, dtype=np.dtype("uint16")):

    tcalc = time.time()

    with h5.File(fname, "w", driver="stdio") as f:
        t0 = 0
        last_chunk = -1
        while t0 < nframes:
            chunk = dataQ.get()
            chunk_size = chunk[1].shape[0]
            if t0 == 0:
                f.require_dataset(
                    "data",
                    (nframes, *chunk[1].shape[1:]),
                    dtype=dtype,
                    compression="gzip",
                )
            chunk_diff = chunk[0] - last_chunk
            if chunk_diff != 1:
                raise IOError(
                    "Chunks have been read in wrong order: chunk index difference is % and not 1."
                    % chunk_diff
                )
            last_chunk, f["/data"][t0 : t0 + chunk_size, ...] = chunk
            t0 += chunk_size
            progress(t0, nframes)
        progress(1, 1)

    # END OF MAIN LOOP put results to output queue
    tcalc = time.time() - tcalc
    print("Elapsed time: %.2fs" % tcalc)


def h5_update_meta(sid, meta, fname):

    with h5.File(fname, "r+", driver="stdio") as f:
        for att in meta.columns:
            f["/data"].attrs[att] = meta.loc[sid, att]


def to_h5(
    obj, sid, filename, nprocs=4, chunk_size=100, dtype=np.dtype("uint16"), **kwargs
):

    print(
        "\n\n#### Writing Data to HDF5 File ####\nSeries: {} in folder {}\n".format(
            sid, obj.meta.loc[sid, "datdir"]
        )
    )

    read_data_opt = {
        "dtype": dtype,
        "nprocs": 1,
        "verbose": False,
    }
    read_data_opt.update(kwargs)

    fmax = obj.meta.loc[sid, "nframes"]
    chunks = [
        np.arange(i * chunk_size, min((i + 1) * chunk_size, fmax))
        for i in range(np.ceil(fmax / chunk_size).astype(np.int32))
    ]

    print("Using {} processes to read data.".format(nprocs))
    MyManager.register(
        "PriorityQueue", PriorityQueue
    )  # Register a shared PriorityQueue
    m = Manager()
    dataQ = m.PriorityQueue(nprocs)
    indxQ = m.PriorityQueue()
    # dataQ = mp.Queue(nread_procs)
    # indxQ = mp.Queue()
    read_data_opt["dataQ"] = dataQ
    read_data_opt["indxQ"] = indxQ
    read_data_opt["method"] = "queue_chunk"

    for i, chunk in enumerate(chunks):
        indxQ.put((i, chunk))

    lock = 0
    if "h5" in obj.fmtstr:
        lock = mp.Lock()
        read_data_opt["lock"] = lock

    procs = []
    for ip in range(nprocs):
        procs.append(
            mp.Process(target=obj.get_series, args=(sid,), kwargs=read_data_opt)
        )
        procs[ip].start()
        if ip == 0:
            time.sleep(2)

    h5_dump(filename, fmax, dataQ)
    h5_update_meta(sid, obj.meta, filename)

    dataQ.task_done()
    indxQ.task_done()

    # stopping processes
    for ip in range(nprocs):
        procs[ip].join()

    # closing queues
    # dataQ.close()
    # dataQ.join_thread()
    # indxQ.close()
    # indxQ.join_thread()

    print("Dataset saved to %s." % filename)
