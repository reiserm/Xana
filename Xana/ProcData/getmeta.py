import h5py
import hdf5plugin
import numpy as np


def get_attrs_from_dict(obj, meta):
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))

    def get_attr(header, p):
        try:
            attr = p[1](header[p[0]])
        except KeyError:
            attr = 0
        return attr

    nfiles = len(meta["master"])
    p = obj.attributes
    init_meta(p)
    for i, filename in enumerate(meta.copy()["master"]):
        header = obj.get_header(str(filename))
        for key, value in p.items():
            meta[key][i] = get_attr(header, value)


def get_header_h5(*args, **kwargs):
    return 0


def get_attrs_h5(obj, meta):
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))

    def get_attr(f, p, key):
        if p[0] == obj.h5opt["data"]:
            attr = f[p[0]].attrs[key]
        else:
            if p[0]:
                attr = f[p[0]]
                if attr.dtype.kind == "V":
                    attr = np.mean(attr.value["value"]).astype(np.dtype(p[1]))
                elif attr.dtype.kind == "f":
                    attr = attr[()]  # .astype(np.dtype(p[1]))
                elif attr.dtype.kind == "u":
                    attr = attr[()]  # .astype(np.dtype(p[1]))
                elif attr.dtype.kind == "i":
                    attr = attr[()]
            else:
                attr = f[obj.h5opt["data"]].shape[0]
        return attr

    nfiles = len(meta["master"])
    p = obj.attributes
    init_meta(p)
    for i, filename in enumerate(meta.copy()["master"]):
        with h5py.File(filename, "r", driver=obj.h5opt["driver"]) as f:
            for key, value in p.items():
                meta[key][i] = get_attr(f, value, key)


"""
AGIPD Methods work in progress ...
"""

# class Pulse:
#
#     def __init__(self,):
#         pass
#
# class Train(Pulse):
#
#     def __init__(self,):
#         self.id = None
#         self.npulses = None
#         self.pulse_ids = None
#
# class Run(Train):
#
#     def __init__(self, runid, rundir, attrdict):
#         self.id = runid
#         self.rundir = rundir
#         self.attrdict = attrdict
#         self.ntrains = None
#         self.train_ids = None
#
#     def get_(self):
#         pass
#
# class MidMetadata(Run):
#
#     def __init__(self, *uid, depth=None, datdir='./',):
#         self.datdir = os.path.abspath(datdir)
#         if depth is None:
#             depth = len(uid)
#         self.depth = depth
#
#
#
# def get_attrs_agipd(obj, meta, depth, uid):
#
#     def init_meta(p):
#         for key, value in p.items():
#             meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))
#
#     def get_attr(f, p, key):
#         if p[0] == obj.h5opt['data']:
#             attr = f[p[0]].attrs[key]
#         else:
#             if p[0]:
#                 attr = f[p[0]]
#                 if attr.dtype.kind == 'V':
#                     attr = np.mean(attr.value['value']).astype(np.dtype(p[1]))
#                 elif attr.dtype.kind == 'f':
#                     attr = attr[()]#.astype(np.dtype(p[1]))
#                 elif attr.dtype.kind == 'u':
#                     attr = attr[()]#.astype(np.dtype(p[1]))
#                 elif attr.dtype.kind == 'i':
#                     attr = attr[()]
#             else:
#                 attr = f[obj.h5opt['data']].shape[0]
#         return attr
#
#     f_files = getfiles(obj.datdir, obj.suffix, obj.fastname)
#     s_files = getfiles(obj.datdir, obj.suffix, obj.slowname)
#     print(f_files,s_files)
#
#     p = obj.attributes
#     s = obj.slowdata
#     init_meta(p)
#     for i, m in enumerate(meta.copy()['master']):
#         filename = obj.datdir + m
#         with h5py.File(filename, 'r', driver=obj.h5opt['driver']) as f:
#             for key, value in p.items():
#                 meta[key][i] = get_attr(f, value, key)


def common_mode_from_hist(im, searchoffset=50):
    bins = np.arange(-searchoffset, searchoffset)
    hist, _ = np.histogram(im, bins)
    CM = np.where(hist == hist.max())[0][0] - searchoffset
    return CM  # has to be subtracted from the tail
