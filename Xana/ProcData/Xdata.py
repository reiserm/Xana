import os
import sys
import re
import numpy as np
import pandas as pd
from .Xfmt import Xfmt
from .EdfMethods import loadedf
from .to_h5 import to_h5
from ..misc.makemask import masker

class Xdata(Xfmt):

    def __init__(self, **kwargs):
        super().__init__(kwargs.pop('fmtstr', ''))
        self.maskfile = kwargs.pop('maskfile', '')
        self.datdir = kwargs.pop('datdir', '')
        self.mask = self.get_mask()
        self.filelist = None
        self.masters = []
        self.headers = []
        self.meta  = None
        self.serieslist = []
        self.series_ids = None
        
    def connect(self, datdir):
        if datdir == '':
            raise ValueError('Data directory is empty. Use valid data directory.')
        self.datdir = os.path.abspath(datdir) + '/'
        self.filelist = self.get_filelist(self.datdir)
        start = len(self.masters)
        self.get_masters()
        self.get_headers(start)
        self.get_serieslist(start)
        self.get_meta(start)

    def get_filelist(self, datdir):
        return self.getfiles(datdir, self.suffix, self.numfmt)

    def get_masters(self):
        master = re.compile(self.masterfmt + r"\."+ self.suffix)
        masters = [master.search(x).group().split('/')[-1] for x in self.filelist if bool(master.search(x))]
        self.masters.extend(masters)

    def get_headers(self, start=0):
        headers = []
        for m in self.masters[start:]:
            self.headers.append(self.get_header(self.datdir+m))
        
    def get_meta(self, start=0):
        meta = {'series':self.series_ids, 'master':self.masters[start:], 'datdir':[self.datdir]*len(self.masters[start:])}
        self.get_attributes(self, meta)
        meta = pd.DataFrame.from_dict(meta)
        meta = meta.reindex(columns=['series'] + list([a for a in meta.columns if a not in ['series', 'master', 'datdir']])
                                              + ['master', 'datdir'])
        if self.meta is None:
            self.meta = meta
        else:
            self.meta = self.meta.append(meta, ignore_index=True)
            self.meta.drop_duplicates(inplace=True)
            self.meta.reset_index(drop=True, inplace=True)        
            
    def get_serieslist(self, start=0):
        tmp = self.files2series(self.filelist, self.masters[start:], self.seriesfmt,)
        self.serieslist.extend(tmp[0])
        self.series_ids = tmp[1]
        self.filelist = None
    
    def get_series(self, series_id, **kwargs):
        return self.load_data_func(self.serieslist[series_id], xdata=self, **kwargs)

    def get_image(self, imgn, series_id=0, **kwargs):
        return self.load_data_func(self.serieslist[series_id], first=(imgn,),
                                   last=(imgn+1,), **kwargs)[0]

    def get_mask(self):
        self.maskfile = os.path.abspath(self.maskfile)
        if self.maskfile.endswith('edf'):
            mask = loadedf(self.maskfile)
        elif self.maskfile.endswith('npy'):
            mask = np.load(self.maskfile)
        else:
            print('Mask file not found. Continuing without mask.')
            mask = None
        return mask

    def masker(self, Isaxs, **kwargs):
        mask = kwargs.get('mask', self.mask)
        masker(Isaxs, mask)

    def to_h5(self, series_id, filename, **kwargs):
        to_h5(self, series_id, filename, **kwargs)

    
        
