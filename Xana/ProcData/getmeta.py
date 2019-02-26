import h5py
import numpy as np
import os
import re
from os.path import isfile


#######################################
#--- get information on data series ---#
#######################################    

def getfiles(datdir, suffix, numfmt='(_\d)*'):
    check_suffix = re.compile(suffix)
    filelist = [ os.path.abspath(datdir + item) for item in os.listdir(datdir)
                 if os.path.isfile(os.path.join(datdir, item))
                 and bool(check_suffix.search(item))]
    filelist = sorted(filelist,
                      key=lambda x: int(''.join(re.findall( numfmt ,x.split('/')[-1]))))
    return filelist

def files2series(filelist, masters, seriesfmt,):
    series = []
    series_id = np.empty(len(masters), dtype=np.int32)
    find_seriesid = re.compile(seriesfmt)
    for i,m in enumerate(masters):
        idstr = find_seriesid.search(m).group()
        series_id[i] = int(idstr)
        nblocks = len(re.findall('(_\d{4,})', m))
        searchstr = idstr + '.*(_\d{{4,}}){{{}}}'.format(nblocks-1)
        series.append([x for x in filelist if re.search(searchstr, x) is not None])
    return series, series_id

def get_attrs_from_dict(obj, meta, *args, **kwargs):
            
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))
            
    def get_attr(header, p):
        try:
            attr = p[1](header[p[0]])
        except:
            attr = 0
        return attr

    nfiles = len(meta['master'])
    p = obj.attributes
    init_meta(p)
    for i, m in enumerate(meta.copy()['master']):
        filename = obj.datdir + m
        header = obj.get_header(filename)
        for key, value in p.items():
            meta[key][i] = get_attr(header, value)
                
def get_header_h5(*args, **kwargs):
    return 0

def get_attrs_h5(obj, meta, *args, **kwargs):
            
    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))
            
    def get_attr(f, p, key):
        if p[0] == obj.h5opt['data']:
            attr = f[p[0]].attrs[key]
        else:
            if p[0]:
                attr = f[p[0]]
                if attr.dtype.kind == 'V':
                    attr = np.mean(attr.value['value']).astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'f':
                    attr = attr[()]#.astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'u':
                    attr = attr[()]#.astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'i':
                    attr = attr[()]
            else:
                attr = f[obj.h5opt['data']].shape[0]
        return attr

    nfiles = len(meta['master'])
    p = obj.attributes
    init_meta(p)
    for i, m in enumerate(meta.copy()['master']):
        filename = obj.datdir + m
        with h5py.File(filename, 'r', driver=obj.h5opt['driver']) as f:
            for key, value in p.items():
                meta[key][i] = get_attr(f, value, key)

                
'''
AGIPD Methods
'''

class Pulse:

    def __init__(self,):
        pass

class Train(Pulse):

    def __init__(self,):
        self.id = None
        self.npulses = None
        self.pulse_ids = None

class Run(Train):

    def __init__(self, runid, rundir, attrdict):
        self.id = runid
        self.rundir = rundir
        self.attrdict = attrdict
        self.ntrains = None
        self.train_ids = None

    def get_(self):
        pass

class MidMetadata(Run):

    def __init__(self, *uid, depth=None, datdir='./',):
        self.datdir = os.path.abspath(datdir)
        if depth is None:
            depth = len(uid)
        self.depth = depth
        
    

def get_attrs_agipd(obj, meta, depth, uid):

    def init_meta(p):
        for key, value in p.items():
            meta[key] = np.empty(nfiles, dtype=np.dtype(value[1]))
            
    def get_attr(f, p, key):
        if p[0] == obj.h5opt['data']:
            attr = f[p[0]].attrs[key]
        else:
            if p[0]:
                attr = f[p[0]]
                if attr.dtype.kind == 'V':
                    attr = np.mean(attr.value['value']).astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'f':
                    attr = attr[()]#.astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'u':
                    attr = attr[()]#.astype(np.dtype(p[1]))
                elif attr.dtype.kind == 'i':
                    attr = attr[()]
            else:
                attr = f[obj.h5opt['data']].shape[0]
        return attr

    f_files = getfiles(obj.datdir, obj.suffix, obj.fastname)
    s_files = getzafiles(obj.datdir, obj.suffix, obj.slowname)
    print(f_files,s_files)

    p = obj.attributes
    s = obj.slowdata
    init_meta(p)
    for i, m in enumerate(meta.copy()['master']):
        filename = obj.datdir + m
        with h5py.File(filename, 'r', driver=obj.h5opt['driver']) as f:
            for key, value in p.items():
                meta[key][i] = get_attr(f, value, key)

def common_mode_from_hist(im, searchoffset=50):                        
    bins = np.arange(-searchoffset,searchoffset) 
    hist, _ = np.histogram(im,bins)
    CM = np.where(hist==hist.max())[0][0] - searchoffset
    return CM # has to be subtracted from the tail
