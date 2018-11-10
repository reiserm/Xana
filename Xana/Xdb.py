import sys
import os
import re
import numpy as np
import pickle
import pandas as pd

from Decorators import Decorators
from misc.xsave import save_result, make_filename


class Xdb:
    """Data base for handling analysis results
    """

    def __init__(self, dbfile=None, db=None, **kwargs):
        self.db = db
        self.dbfile = dbfile
        if dbfile is not None:
            self.load_db(dbfile)
        self.savdir = None
        self.setupfile = None

    def __str__(self):
        return ('Xana Instance\n' +
                "savdir: {}\n" +
                "sample name: {}\n" +
                "database file: {}\n" +
                'setup file: {}\n').format(self.savdir, self.sample,
                                           self.dbfile, self.setupfile)

    def __repr__(self):
        return self.__str__()

    # Data Base
    def load_db(self, dbfile=None, init=False, **kwargs):
        path, fname = make_filename(self, 'dbfile', dbfile)
        dbfile = path + fname
        print('Try loading database:\n\t{}'.format(dbfile))
        try:
            self.db = pickle.load(open(dbfile, 'rb'))
            self.dbfile = dbfile
            print('Successfully loaded database')
        except OSError:
            print('\t...loading database failed.')
            if init:
                self.dbfile = self.savdir + os.path.split(dbfile)[-1]
                print(
                    'Analysis database not found.\nInitialize database...'.format(dbfile))
                self.init_db(dbfile, **kwargs)

    def init_db(self, dbfile=None, handle_existing='raise'):
        if self.meta is not None:
            meta_names = list(self.meta.columns)
        else:
            meta_names = []
        names = ['use', 'sample', 'analysis', *meta_names,
                 'mod', 'savname', 'savfile', 'setupfile', 'comment']
        tmp_db = pd.DataFrame(columns=names)
        self.db = tmp_db
        path, fname = make_filename(self, 'dbfile', dbfile)
        self.dbfile = path + fname
        self.save_db(handle_existing=handle_existing)

    def add_db_entry(self, series_id, savfile, method):
        self.db.reset_index(inplace=True, drop=True)
        savname = savfile.split('/')[-1]
        dbn = self.db.shape[0]
        entry = [True, self.sample, method, *self.meta.loc[series_id],
                 pd.datetime.today(), savname, savfile, self.setupfile, ""]
        self.db.loc[dbn] = entry
        self.save_db(handle_existing='overwrite')

    @Decorators.input2list
    def discard_entry(self, db_id, save=True):
        discard = []
        for i in db_id:
            datdir = self.db.loc[i, 'datdir']
            series = self.db.loc[i, 'series']
            sample = self.db.loc[i, 'sample']
            discard.append(self.db[(self.db['datdir'].str.match(datdir))
                                   & (self.db['series'] == series)
                                   & (self.db['sample'].str.match(sample))].index.values)
        self.db.loc[np.unique(np.hstack(discard)), 'use'] = False
        if save:
            self.save_db(handle_existing='overwrite')

    def save_db(self, filename=None, handle_existing='raise'):
        savdir, dbfile = make_filename(self, 'dbfile', filename)
        self.dbfile = save_result(
            self.db, 'Analysis', savdir, dbfile, handle_existing=handle_existing)

    def append_db(self, dbfile, check_duplicates=True):
        if type(dbfile) == str:
            path, fname = make_filename(self, filename=dbfile)
            dbfile = path + fname
            db = pickle.load(open(dbfile, 'r+b'))
        self.db = self.db.append(db, ignore_index=True)

        if len(self.db[self.db['use'] == False]) and check_duplicates:
            self.discard_entry(
                self.db[self.db['use'] == False].index.values, save=False)

        self.db.drop_duplicates(inplace=True)
        self.db.reset_index(drop=True, inplace=True)

    def get_item(self, item):
        if type(item) == str:
            return pickle.load(open(item, 'rb'))
        else:
            return pickle.load(open(self.db.loc[item]['savfile'], 'rb'))

    @Decorators.input2list
    def rm_db_entry(self, db_id, rmfile=False):
        for i in db_id:
            if rmfile:
                try:
                    os.remove(self.db.loc[i]['savfile'])
                except FileNotFoundError:
                    print('File not found.')
            self.db = self.db.drop(i)
        self.save_db()
