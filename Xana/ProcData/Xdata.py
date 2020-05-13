import os
import re
import pandas as pd
import numpy as np
from .Xfmt import Xfmt
from .to_h5 import to_h5
from ..misc.makemask import masker
import warnings

class Xdata(Xfmt):
    '''Class to get metadata information on datasets.
    '''

    def __init__(self, datdir=None, fmtstr=None):
        super().__init__(fmtstr)
        self.datdir = datdir
        self._files = None
        self._masters = []
        self._headers = []
        self.meta  = []
        self._meta_save = None
        self._series = []
        self._series_ids = None

    def connect(self, datdir, addfirstnlast=True, checksubseries=True, nframesfromfiles=False):
        """Finds datasets in the directory given by :code:`datdir`.

        Args:
            datdir (str): data directory that contains data files.
        """
        if not os.path.isdir(datdir):
            warnings.warn('Data directory does not exist. Use valid data directory.')
            return
        self.datdir = os.path.abspath(datdir) + '/'

        if isinstance(self.fmtstr, str) and 'agipd' not in self.fmtstr:
            self._get_files(self.datdir)
            self._get_masters()
            self._get_headers()
            self._files2series()
        self._get_meta(addfirstnlast, checksubseries, nframesfromfiles)

    def _get_files(self, datdir,):
        check_suffix = re.compile(self.suffix)
        files = [os.path.abspath(datdir + item) for item in os.listdir(datdir)
                    if os.path.isfile(os.path.join(datdir, item))
                    and bool(check_suffix.search(item))]
        self._files = sorted(files,
                          key=lambda x: int(''.join(re.findall(self.numfmt, x.split('/')[-1]))))

    def _get_masters(self):
        master = re.compile(self.masterfmt + r"\."+ self.suffix)
        masters = [master.search(x).group().split('/')[-1] for x in self._files
                   if bool(master.search(x))]
        self._masters = masters

    def _get_headers(self,):
        headers = []
        for m in self._masters:
            headers.append(self.get_header(self.datdir+m))
        self._header = headers

    def _files2series(self,):
        series = []
        series_id = []
        find_seriesid = re.compile(self.seriesfmt)
        for i, m in enumerate(self._masters):
            idstr = find_seriesid.search(m).group()
            series_id.append(int(idstr))
            nblocks = len(re.findall('(_\d{4,})', m))
            searchstr = idstr + '.*(_\d{{4,}}){{{}}}'.format(nblocks - 1)
            series.append([x for x in self._files if re.search(searchstr, x) is not None])
        self._series_ids = np.asarray(series_id, dtype='int32')
        self._series.extend(series)

    def _get_meta(self, addfirstnlast=True, checksubseries=True, nframesfromfiles=False):
        meta = {'series':self._series_ids, 'master':self._masters,
                'datdir':[self.datdir]*len(self._masters)}
        self.get_attributes(self, meta,)
        meta = pd.DataFrame.from_dict(meta)
        meta = meta.reindex(columns=['series']
                            + list([a for a in meta.columns
                                  if a not in ['series', 'master', 'datdir']])
                            + ['master', 'datdir'])
        if nframesfromfiles:
            for idx, row in meta.iterrows():
                row['nframes'] = len(self._series[idx])
                meta.loc[idx] = row

        if addfirstnlast:
            meta.insert(5, 'last', int(0))
            meta.insert(5, 'first', int(0))

            for idx, row in meta.iterrows():
                row['first','last'] = (0, int(row['nframes']-1))
                meta.loc[idx] = row
                if checksubseries:
                    tot_img = self.get_series(idx, verbose=False, output='shape')[0]
                    img_per_series = row['nframes']
                    nrow = row.copy()
                    idx_subset = 1
                    while nrow['last'] + 1 - tot_img < 0:
                        if 'subset' not in meta:
                            meta.insert(1, 'subset', int(0))
                        nrow['first'] = img_per_series + nrow['first']
                        nrow['last'] = img_per_series + nrow['last']
                        nrow['subset'] = idx_subset
                        idx_subset += 1
                        meta.loc[meta.shape[0]] = nrow


        if not len(self.meta):
            self.meta = meta
        else:
            self.meta = self.meta.append(meta, ignore_index=True)
            self.meta.drop_duplicates(inplace=True)
            self.meta.reset_index(drop=True, inplace=True)

    def get_series(self, series_id, **kwargs):
        """Reads data.

        Args:
            series_id (int): index of the dataset in the :code:`Xana.meta` table.
            **kwargs: arguments passed to the data loader.

        Returns:
            np.ndarray: if method is :code:`full` or tuple of average images and variance
            if method is :code:`average`.
        """
        if 'subset' in self.meta:
            nf = self.meta.loc[series_id, 'nframes']
            first = self.meta.loc[series_id, 'first']
            last = self.meta.loc[series_id, 'last']
            kwargs['first'] =  kwargs.get('first', first) % nf + first
            kwargs['last'] = kwargs.get('last', last) % nf + first
            series_id = self.meta[(self.meta['series']==self.meta.loc[series_id, 'series'])
                                    & (self.meta['subset']==0)].index.values[0]
        return self.load_data_func(self._series[series_id], xdata=self, **kwargs)

    def get_image(self, series_id, imgn=0, **kwargs):
        """Returns a single image of a dataset.

        Args:
            series_id (int): Index of the dataset in the :code:`Xana.meta` table.
            imgn (int, optional): Index of image starting with 0. Defaults 0.
            **kwargs: arguments passed to the data loader.

        Returns:
            np.ndarray: The 2D image. Shape depends on the detector.
        """
        return self.load_data_func(self._series[series_id], xdata=self, first=(imgn,),
                                   last=(imgn+1,), **kwargs)[0]

    def masker(self, Isaxs, **kwargs):
        mask = kwargs.get('mask', self.setup.mask)
        masker(Isaxs, mask)

    def to_h5(self, series_id, filename, **kwargs):
        """Convert dataset to h5.

        Built in to convert single edf series from ID10 or ID02.

        Args:
            series_id (int): Index of the dataset in the :code:`Xana.meta` table.
            filename (str): Filename of the HDF5 file.
        """
        to_h5(self, series_id, filename, **kwargs)
