import h5py
import numpy as np
import os
import re
from os.path import isfile

# from EdfMethods import loadedf, headeredf


#######################################
# --- get information on hdf5 series ---#
#######################################

# --- older version of gertfiles; just kept here in case needed for p10 experiment ---
# def getfiles(datdir, suffix, numfmt='(_\d)*'):
#     filelist = [ os.path.abspath(datdir + item) for item in os.listdir(datdir)
#                if os.path.isfile(os.path.join(datdir, item)) and item.endswith(suffix) ]
#     filelist = sorted(filelist, key=lambda x: int(''.join(re.findall( numfmt ,x))))
#     return filelist


def getfiles(datdir, suffix, numfmt="(_\d)*"):
    check_suffix = re.compile(suffix)
    filelist = [
        os.path.abspath(datdir + item)
        for item in os.listdir(datdir)
        if os.path.isfile(os.path.join(datdir, item))
        and bool(check_suffix.search(item))
    ]
    filelist = sorted(
        filelist, key=lambda x: int("".join(re.search(numfmt, x).group(0).split("_")))
    )
    return filelist


def files2series(
    filelist,
    masters,
    seriesfmt,
):
    series = []
    series_id = np.empty(len(masters), dtype=np.int32)
    find_seriesid = re.compile(seriesfmt)
    for i, m in enumerate(masters):
        idstr = find_seriesid.search(m).group()
        series_id[i] = int(idstr)
        nblocks = len(re.findall("(_\d{4,})", m))
        searchstr = idstr + ".*(_\d{{4,}}){{{}}}".format(nblocks - 1)
        series.append([x for x in filelist if re.search(searchstr, x) is not None])
    return series, series_id


def get_attr_from_dict(obj, meta):
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))

    def get_attr(header, p):
        try:
            attr = p[1](header[p[0]])
        except:
            attr = 0
        return attr

    nfiles = len(meta["master"])
    p = obj.attributes
    init_meta(p)
    for i, m in enumerate(meta.copy()["master"]):
        filename = obj.datdir + m
        header = headeredf(filename)
        for key, value in p.items():
            meta[key][i] = get_attr(header, value)


def get_attributes(filename, c, obj):
    p = obj.attributes[c]
    f = headeredf(filename)
    try:
        attr = p[1](f[p[0]])
    except:
        attr = 0
    return attr


def get_header_h5(filename):
    # f = h5py.File(filename, 'r', driver='stdio')
    # header = dict(f['/entry/instrument/detector/'])
    # f.close()
    return 0


def get_attributes_h5(obj, meta):
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))

    def get_attr(f, p):
        if p[0] == obj.h5opt["data"]:
            attr = f[p[0]].attrs[c]
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
    for i, m in enumerate(meta.copy()["master"]):
        filename = obj.datdir + m
        with h5py.File(filename, "r", driver=obj.h5opt["driver"]) as f:
            for key, value in p.items():
                meta[key][i] = get_attr(f, value)


def common_mode_from_hist(im, searchoffset=50):
    bins = np.arange(-searchoffset, searchoffset)
    hist, _ = np.histogram(im, bins)
    CM = np.where(hist == hist.max())[0][0] - searchoffset
    return CM  # has to be subtracted from the tail
