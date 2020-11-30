import h5py
import hdf5plugin
import numpy as np
import multiprocessing as mp
from ..XpcsAna.xpcsmethods import mat2evt
from ..misc.progressbar import progress
from . import EdfMethods as edf
from . import CbfMethods as cbf


def get_case(detector):
    """select the way data are read"""
    if detector in [
        "id10_eiger_single_edf",
        "pilatus_single_cbf",
        "id02_eiger_single_edf",
    ]:
        case = 0
    elif detector in [
        "p10_eiger_h5",
        "converted_h5",
        "lambda_nxs",
        "ebs_id02_h5",
        "ebs_id10_h5",
    ]:
        case = 1
    elif detector == "xcs_cspad_h5":
        case = 2
    elif detector == "id02_eiger_multi_edf":
        case = 3
    else:
        case = -1
    return case


# def evt2prob( events, qroi, nbins ):

#     nq = len(events)
#     prob = np.empty((nq,nbins+1), np.float32)
#     for iq in range(nq):
#         counts = np.unique(events[iq], return_counts=1)[1]
#         tmp = np.histogram(counts, nbins-1, range=(1,nbins-1))[0]
#         npix = qroi[iq][0].size
#         prob[iq,:] = np.hstack((counts.sum(),npix-counts.size,tmp))/npix
#     return prob


def get_firstnlast(first, last, nf, dim):

    nf = nf - 1

    if last is None and first is None:
        last = (nf, *dim)
        first = (0,) * len(last)
    elif last is not None and first is None:
        if isinstance(last, (int, np.integer)):
            last = (last,)
        if len(last) == 1:
            last = (min([last[0], nf]), *dim)
        else:
            last = [min([last[0], nf]), *last[1:]]
        first = (0,) * len(last)
    elif last is None and first is not None:
        if isinstance(first, (int, np.integer)):
            first = (first,)
        if len(first) > 1:
            last = (nf, *dim)
        else:
            first = (first[0], 0, 0)
            last = (nf, *dim)
    elif last is not None and first is not None:
        if isinstance(first, (int, np.integer)):
            first = (first, 0, 0)
        else:
            first = (first[0], 0, 0)
        if isinstance(last, (int, np.integer)):
            last = (last, *dim)
        else:
            last = (last[0], *dim)
    elif last is not None and first is not None and len(last) == len(first) == 2:
        first = (0, first[1], first[0])
        last = (nf, last[1], last[0])
    return first, last


def alock(lock):
    """acquire lock"""
    if lock:
        lock.acquire()
    else:
        pass


def rlock(lock):
    """release lock"""
    if lock:
        lock.release()
    else:
        pass


def rearrange_tiles(arr, func):

    if arr.ndim > 3:
        for i in range(arr.shape[0]):
            if i == 0:
                tmp = func(arr[i])
                arranged_arr = np.empty((arr.shape[0], *tmp.shape))
                arranged_arr[i] = tmp
            else:
                arranged_arr[i] = func(arr[i])
        del arr
        arr = arranged_arr
    else:
        arr = func(arr)
    return arr


# -----------------------------------------------------------------------
# data format classes to provide specific functions to read data and get
# information on dataset shape
class dataset:
    """
    Class that provides all meta information, options,
    correction functions for data loading
    """

    def __init__(self, opt):

        self.__dict__.update(opt)
        self.shape = None
        self.dstream = None
        self.variance = None
        self.chunk = None
        self.imgpf = None

    def update_shape(self, nimg, dim):

        self.shape = (nimg, *dim)

    def init_output(
        self,
    ):

        if self.method == "full":
            self.dstream = np.empty(self.shape, dtype=self.dtype)
        elif self.method == "average":
            pass
        elif self.method == "queue_chunk":
            self.dstream = self.chunk
        elif self.method == "events":
            if self.qroi is None:
                self.qroi = [self.mask]
                print("Qrois not defined. Using all pixels that are not masked.")
            self.dstream = [[] for i in range(len(self.qroi) + 1)]
        else:
            pass

    def calc_mean(self, weighted=False):

        self.dstream = np.ma.masked_array(self.dstream)
        self.variance = np.ma.masked_array(self.variance)
        m, v = np.ma.average(
            self.dstream, axis=0, weights=1 / self.variance, returned=True
        )
        v = 1 / v
        self.variance = v.data
        if weighted:
            self.dstream = m.data
        else:
            self.dstream = self.dstream.mean(0).data

    def process_chunk(self):

        arr = self.chunk.astype(self.dtype, copy=False)

        for correction in self.corrections:
            if correction == "commonmode":
                for matr in arr:
                    for tile in matr:
                        tile -= self.commonmode(tile)

            elif correction == "dark":
                arr -= self.dark[None, ...]

            # elif correction == 'mask':
            #     ind = (...,*mask)
            #     arr[ind] = mask_value

            elif correction == "filter_value":
                arr[arr != self.filter_value] = 0

            elif correction == "dropletize":
                arr = self.dropletize(arr, self.dropopt)

        if "events" in self.method:
            # store mean intensity of frames in first list element
            if self.mask is not None:
                ind = (..., *self.mask)
                m = [
                    arr[ind].mean(1),
                ]
            else:
                m = [
                    arr.mean((1, 2)),
                ]

            # extend the dstream list by the intensities in the pixels
            # in the previously defined qrois
            for qi in range(len(self.qroi)):
                ind = (..., *self.qroi[qi])
                m.append(list(mat2evt(arr[ind])))

            arr = m

        self.chunk = arr

    def prepare_output(self):

        if self.output == "original":
            pass
        else:
            if "2d" in self.output:
                if self.cond_rearrange_tiles():
                    self.dstream = rearrange_tiles(self.dstream, self.arrange_tiles)
                    if self.variance is not None:
                        self.variance = rearrange_tiles(
                            self.variance, self.arrange_tiles
                        )
                if "sec" in self.output:
                    if self.cond_section():
                        qsec = self.qsec
                        self.dstream = self.dstream[
                            ...,
                            qsec[0][0] : qsec[1][0] + 1,
                            qsec[0][1] : qsec[1][1] + 1,
                        ]
                        if self.variance is not None:
                            self.variance = self.variance[
                                ...,
                                qsec[0][0] : qsec[1][0] + 1,
                                qsec[0][1] : qsec[1][1] + 1,
                            ]
                if self.dstream.shape[0] == 1:
                    self.dstream = np.squeeze(self.dstream)

    def cond_rearrange_tiles(self):

        if self.method == "average":
            if self.dstream.ndim > 2:
                return True
        elif self.method == "full":
            if self.dstream.ndim > 3:
                return True
        elif self.method == "queue_chunk":
            if self.chunk.ndim > 3:
                return True
        else:
            return False

    def cond_section(self):

        qsec = self.qsec
        if qsec is not None:
            if self.dstream.shape[-2:] != (
                qsec[1][0] - qsec[0][0] + 1,
                qsec[1][1] - qsec[0][1] + 1,
            ):
                return True
            else:
                return False
        else:
            return False


class hdf5(dataset):
    def __init__(self, masterfile, opt, use_chunks=True):

        super().__init__(opt)
        self.masterfile = masterfile
        self.use_chunks = use_chunks

    def get_shape(self):

        alock(self.lock)

        with h5py.File(self.masterfile, "r", driver=self.driver) as f:
            if self.extlinks:
                data_links = [
                    f[self.datapath].name + "/" + name for name in f[self.datapath]
                ]
                datapath = data_links
            else:
                datapath = [
                    self.datapath,
                ]
            dim = f[datapath[0]].shape
            nf = 0
            # determine number of images per file
            self.imgpf = dim[0]
            for d in datapath:
                nf += f[d].shape[0]

        rlock(self.lock)
        self.datapath = datapath
        self.shape = (nf, *dim[1:])
        return self.shape

    def load_chunk(self, indx=None):

        alock(self.lock)
        datapath = self.datapath[int(indx[0] // self.imgpf)]
        indx = (indx % self.imgpf).astype("int32")
        qsec = self.qsec
        with h5py.File(self.masterfile, "r", driver=self.driver) as f:
            if indx is not None and qsec is not None:
                if len(f[datapath].shape) > 1:
                    arr = f[datapath][
                        indx[0] : indx[-1] + 1,
                        qsec[0][0] : qsec[1][0] + 1,
                        qsec[0][1] : qsec[1][1] + 1,
                    ]
                else:
                    arr = f[datapath][indx[0] : indx[-1] + 1]
            elif indx is not None:
                arr = f[datapath][indx[0] : indx[-1] + 1]
            else:
                arr = f[datapath][...]

        rlock(self.lock)

        self.chunk = arr

    def start_reading_data(self):

        return None

    def stop_reading_data(self):

        return None


class sglimgfmt(dataset):
    def __init__(self, masterfile, datafiles, opt, use_chunks=True):

        super().__init__(opt)
        self.masterfile = str(masterfile)
        if self.masterfile.endswith("edf") or self.masterfile.endswith("edf.gz"):
            self.get_image = edf.loadedf
        elif self.masterfile.endswith("cbf"):
            self.get_image = cbf.loadcbf
        else:
            raise KeyError("Data format neither edf nor cbf.")
        self.datafiles = datafiles
        self.use_chunks = use_chunks
        self.procs = []
        self.in_queue = []
        self.out_queue = []

    def get_shape(self):

        dim = self.get_image(self.masterfile).shape
        nf = len(self.datafiles)
        self.shape = (nf, *dim)
        return self.shape

    def load_chunk(self, indx):

        self.chunk = np.empty((len(indx), *self.shape[1:]), self.dtype)
        for ip in range(self.nprocs):
            self.in_queue[ip].put(
                self.datafiles[indx[0] + ip : indx[-1] + 1 : self.nprocs]
            )

        ip = 0
        for i in range(len(indx)):
            self.chunk[i] = self.out_queue[ip].get()
            ip += 1
            if ip == self.nprocs:
                ip = 0

    def load_proc(self, in_queue, out_queue, qsec=None, dtype=None):

        while True:
            from_queue = in_queue.get()
            if from_queue is None:
                break
            for f in from_queue:
                matr = self.get_image(str(f)).astype(dtype)
                if qsec is not None:
                    matr = matr[
                        qsec[0][0] : qsec[1][0] + 1, qsec[0][1] : qsec[1][1] + 1
                    ]
                out_queue.put(matr)

    def start_reading_data(self):

        for ip in range(self.nprocs):
            self.in_queue.append(mp.Queue(self.nprocs))
            self.out_queue.append(mp.Queue(self.nprocs))
            self.procs.append(
                mp.Process(
                    target=self.load_proc,
                    args=(
                        self.in_queue[ip],
                        self.out_queue[ip],
                    ),
                    kwargs={"qsec": self.qsec, "dtype": self.dtype},
                )
            )
            self.procs[ip].start()

    def stop_reading_data(self):

        for ip in range(self.nprocs):
            self.in_queue[ip].put(None)
            self.in_queue[ip].close()
            self.in_queue[ip].join_thread()
            self.out_queue[ip].close()
            self.out_queue[ip].join_thread()
            self.procs[ip].join()


class multiedf(dataset):
    def __init__(self, masterfile, datafiles, opt, use_chunks=True):

        super().__init__(opt)
        self.masterfile = masterfile
        self.get_image = edf.loadedf
        self.datafiles = datafiles
        self.use_chunks = use_chunks
        self.procs = []
        self.in_queue = []
        self.out_queue = []

    def get_shape(self):

        dim = self.get_image(self.masterfile).shape
        nf = len(self.datafiles)
        self.shape = (nf, *dim)
        return self.shape

    def load_chunk(self, indx):

        self.chunk = np.empty((len(indx), *self.shape[1:]), self.dtype)
        for ip in range(self.nprocs):
            self.in_queue[ip].put(
                self.datafiles[indx[0] + ip : indx[-1] + 1 : self.nprocs]
            )

        ip = 0
        for i in range(len(indx)):
            self.chunk[i] = self.out_queue[ip].get()
            ip += 1
            if ip == self.nprocs:
                ip = 0

    def load_proc(self, in_queue, out_queue, qsec=None, dtype=None):

        while True:
            from_queue = in_queue.get()
            if from_queue is None:
                break
            for f in from_queue:
                matr = self.get_image(f).astype(dtype)
                if qsec is not None:
                    matr = matr[
                        qsec[0][0] : qsec[1][0] + 1, qsec[0][1] : qsec[1][1] + 1
                    ]
                out_queue.put(matr)

    def start_reading_data(self):

        for ip in range(self.nprocs):
            self.in_queue.append(mp.Queue(self.nprocs))
            self.out_queue.append(mp.Queue(self.nprocs))
            self.procs.append(
                mp.Process(
                    target=self.load_proc,
                    args=(
                        self.in_queue[ip],
                        self.out_queue[ip],
                    ),
                    kwargs={"qsec": self.qsec, "dtype": self.dtype},
                )
            )
            self.procs[ip].start()

    def stop_reading_data(self):

        for ip in range(self.nprocs):
            self.in_queue[ip].put(None)
            self.in_queue[ip].close()
            self.in_queue[ip].join_thread()
            self.out_queue[ip].close()
            self.out_queue[ip].join_thread()
            self.procs[ip].join()


# ----------------------
# _*_ MAIN FUNCTION _*_
# ----------------------
def read_data(
    datafiles,
    detector=None,
    last=None,
    first=None,
    step=[1, 1, 1],
    qroi=None,
    qsec=None,
    verbose=True,
    method="full",
    output="2dsection",
    chunk_size=256,
    dtype=np.float32,
    var_weight=False,
    nprocs=1,
    datapath="",
    driver="stdio",
    extlinks=False,
    filter_value=False,
    dropopt=None,
    dropmask=None,
    xdata=None,
    indxQ=None,
    dataQ=None,
    lock=False,
    dark=None,
    commonmode=True,
    dropletize=False,
    mask=False,
    mask_value=-1,
    **kwargs
):

    # ---------------------------------------------
    # Nested Functions only invoked by read_data()
    # ---------------------------------------------
    def make_chunks():
        """
        if data should be read in chunks, this functions createas a list of
        chunked image indices
        """
        if verbose:
            print("Loading data in chunks of {} images.".format(chunk_size))

        # old way
        # chunks = [np.arange(first[0] + i*chunk_size,
        #                     first[0] + min([min([(i + 1) * chunk_size, last[0]]), nimg]),
        #                     step[0])
        #           for i in range(np.ceil(nimg / chunk_size).astype(np.int32))]
        # new chunks
        ind_arange = np.arange(first[0], last[0] + 1)
        bins = np.arange(0, nf, chunk_size)
        digitized = np.digitize(ind_arange, bins)
        chunks = [ind_arange[np.where(digitized == i)] for i in np.unique(digitized)]

        return chunks

    # --------------------
    # Beginning main code
    # --------------------

    # initialize reading options
    options = locals()
    options["corrections"] = []

    if dark is not None:
        if verbose:
            print("Doing dark subtraction.")
        options["dark"] = dark.astype(dtype)
        options["corrections"].append("dark")

    if isinstance(mask, np.ndarray):
        if qsec is not None and "sec" in output:
            mask = mask[qsec[0][0] : qsec[1][0] + 1, qsec[0][1] : qsec[1][1] + 1]
        # options['corrections'].append('mask')

    if xdata is not None:
        try:
            h5opt = xdata.h5opt
        except:
            h5opt = {}
        if commonmode == 1 and "commonmode" in h5opt:
            options["commonmode"] = h5opt["commonmode"]
            options["corrections"].append("commonmode")
            if verbose:
                print("Using commonmode correction.")
        if dropletize == 1 and "dropletize" in h5opt:
            options["dropletize"] = h5opt["dropletize"]
            options["dropopt"] = dropopt
            options["corrections"].append("dropletize")
            if verbose:
                print("Images are dropletized.")
        if "ExternalLinks" in h5opt:
            options["extlinks"] = h5opt["ExternalLinks"]
            if options["extlinks"] and verbose:
                print("H5 file using external links.")
        if datapath == "" and "data" in h5opt:
            options["datapath"] = h5opt["data"]
        if "chunk_size" in h5opt:
            chunk_size = h5opt["chunk_size"]
        if "arrange_tiles" in h5opt and "2d" in output:
            options["arrange_tiles"] = h5opt["arrange_tiles"]
            if verbose:
                print("Rearranging tiles.")
        if "mask" in vars(xdata.setup):
            mask = xdata.setup.mask.copy()
            if qsec is not None and "sec" in output:
                mask = xdata.setup.qsec_mask

    if isinstance(mask, np.ndarray):
        mask = np.where(mask)
        if qsec is not None:
            mask = (mask[0] - qsec[0][0], mask[1] - qsec[0][1])
        options["mask"] = mask
    else:
        mask = None

    if qroi is not None and "sec" in output:
        options["qroi"] = qroi
        for iq in range(len(qroi)):
            options["qroi"][iq] = (qroi[iq][0] - qsec[0][0], qroi[iq][1] - qsec[0][1])

    if "sec" not in output:
        options["qsec"] = None
        qsec = None

    # dcls is the data class that contains all necessary functions
    # to deal with different file formats
    if detector is None and xdata is not None:
        detector = xdata.fmtstr

    case = get_case(detector)
    datafiles = np.array(datafiles)
    masterfile = datafiles[0]

    if case == 0:
        dcls = sglimgfmt(masterfile, datafiles, options)
    elif case in [1, 2]:
        dcls = hdf5(masterfile, options)
    elif case in [3]:
        dcls = multiedf
    else:
        raise ValueError("Case %d not defined." % case)

    shape = dcls.get_shape()
    nf = shape[0]
    dim = shape[1:]

    if output == "shape":
        return shape

    first, last = get_firstnlast(first, last, nf, dim)

    if qsec is not None and len(dim) < 3:
        dim = list(dim)
        dim[-2:] = (qsec[1][0] - qsec[0][0] + 1, qsec[1][1] - qsec[0][1] + 1)

    imgindx = np.arange(first[0], last[0] + 1, step[0])
    nimg = len(imgindx)

    dcls.update_shape(nimg, dim)

    if verbose:
        print("First images is: ", first[0], flush=1)
        print("Last image is: ", last[0], flush=1)

    nargin = nimg
    if dcls.use_chunks:
        if chunk_size is None:
            chunk_size = nimg
        chunks = make_chunks()
        nargin = len(chunks)
        dcls.chunks = chunks

    dcls.init_output()

    # start loading data
    dcls.start_reading_data()

    if method == "full":
        # process full data set
        for i in range(nargin):
            progress(i, max([nargin, 1]))

            dcls.load_chunk(chunks[i])
            dcls.process_chunk()
            dcls.dstream[chunks[i] - first[0]] = dcls.chunk

        dcls.prepare_output()
        progress(1, 1)

    elif method == "events":
        # process full data set
        for i in range(nargin):
            progress(i, max([nargin, 1]))

            dcls.load_chunk(chunks[i])
            dcls.process_chunk()
            for qi in range(len(dcls.dstream)):
                if i == 0:
                    dcls.dstream = dcls.chunk.copy()
                    break
                else:
                    if qi == 0:
                        dcls.dstream[qi] = np.append(dcls.dstream[qi], dcls.chunk[qi])
                    else:
                        for j in range(3):
                            if j == 1:
                                dcls.chunk[qi][j] += chunks[i][0] - first[0]
                            dcls.dstream[qi][j] = np.append(
                                dcls.dstream[qi][j], dcls.chunk[qi][j]
                            )

        progress(1, 1)

    elif method == "average":
        # average data in time dimension
        for i in range(nargin):
            progress(i, max([nargin, 1]))

            dcls.load_chunk(chunks[i])
            dcls.process_chunk()
            if i == 0:
                dcls.dstream = np.empty(
                    (len(chunks), *dcls.chunk.shape[1:]), dtype=dtype
                )
                dcls.variance = dcls.dstream.copy()
            dcls.dstream[i] = dcls.chunk.mean(0)
            dcls.variance[i] = dcls.chunk.var(0)

        dcls.calc_mean(weighted=var_weight)
        dcls.prepare_output()
        progress(1, 1)

    elif method == "queue_chunk":
        # pushing chunks to a queue for external analysis classes
        while not indxQ.empty():
            indx, chunk = indxQ.get()
            dcls.load_chunk(chunk)
            dcls.process_chunk()
            dcls.dstream = dcls.chunk
            dcls.prepare_output()
            dataQ.put((indx, dcls.dstream))

    # done reading data
    dcls.stop_reading_data()

    if method == "queue_chunk":
        return None
    elif method == "average":
        return dcls.dstream, dcls.variance / np.sqrt(nimg)
    else:
        return dcls.dstream
