import os
from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

from .Decorators import Decorators
from .misc.xsave import save_result, make_filename


class Xdb:
    """Data base class for handling analysis results and making them accessible through the
    data interpretation modules.
    """

    def __init__(self, dbfile=None):
        self.db = None
        self.dbfile = dbfile
        if dbfile is not None:
            self.load_db(dbfile)
        self.savdir = None

    # Data Base
    def load_db(self, dbfile=None, init=True, **kwargs):
        dbfile = make_filename(self, "dbfile", dbfile)
        print("Try loading database:\n\t{}".format(dbfile))
        if dbfile.is_file():
            self.db = pickle.load(open(dbfile, "rb"))
            self.dbfile = dbfile
            print("Successfully loaded database")
        elif init and Path(str(self.savdir)).is_dir():
            self.dbfile = self.savdir.joinpath(dbfile.name)
            print("Initialize database...".format(dbfile))
            self.init_db(dbfile, **kwargs)
        else:
            raise ValueError("Loading database failed. Database file not specified or output directory does not exist.")

    def init_db(self, dbfile=None, handle_existing="raise"):
        if self.meta is not None:
            meta_names = list(self.meta.columns)
        else:
            meta_names = []
        names = [
            "use",
            "sample",
            "analysis",
            *meta_names,
            "mod",
            "savname",
            "savfile",
            "setupfile",
            "comment",
        ]
        tmp_db = pd.DataFrame(columns=names)
        self.db = tmp_db
        self.dbfile = make_filename(self, "dbfile", dbfile)
        self.save_db(handle_existing=handle_existing)

    def add_db_entry(self, series_id, savfile, method):
        self.db.reset_index(inplace=True, drop=True)
        savname = savfile.name
        # dbn = self.db.shape[0]
        entry = {
            "use": True,
            "sample": self.sample,
            "analysis": method,
            "mod": datetime.today(),
            "savname": savname,
            "savfile": savfile,
            "setupfile": self.setupfile,
            "comment": "",
        }
        entry.update(
            dict(
                zip(
                    self._meta_save.columns,
                    self._meta_save.loc[series_id],
                )
            )
        )
        entry = pd.DataFrame(entry, index=[0])
        self.db = pd.concat(
            [self.db, entry], join="outer", ignore_index=True, sort=False, copy=False
        )
        self.save_db(handle_existing="overwrite")

    @Decorators.input2list
    def discard_entry(self, db_id, save=True):
        discard = []
        for i in db_id:
            datdir = self.db.loc[i, "datdir"]
            series = self.db.loc[i, "series"]
            sample = self.db.loc[i, "sample"]
            subset = self.db.loc[i].get("subset", np.nan)
            cond = (
                (self.db["datdir"].str.match(str(datdir)))
                & (self.db["series"] == series)
                & (self.db["sample"].str.match(str(sample)))
            )
            if not np.isnan(subset):
                cond &= self.db["subset"] == subset
            discard.append(self.db[cond].index.values)

        if len(discard) == 0:
            print("No entry discarded.")
        else:
            self.db.loc[np.unique(np.hstack(discard)), "use"] = False
            if save:
                self.save_db(handle_existing="overwrite")

    def save_db(self, filename=None, handle_existing="raise"):
        dbfile = make_filename(self, "dbfile", filename)

        if dbfile.is_file() or str(dbfile).endswith(".pkl"):
            folder, filen = dbfile.parent, dbfile.name
        else:
            folder, filen = dbfile, filename

        self.dbfile = save_result(
            self.db, "analysis", folder, filen, handle_existing=handle_existing
        )

    def append_db(self, dbfile, check_duplicates=True):
        if isinstance(dbfile, str):
            path, fname = make_filename(self, filename=dbfile)
            dbfile = path.joinpath(fname)
            if os.path.isfile(dbfile):
                db = pickle.load(open(dbfile, "r+b"))
            else:
                print("File %s does not exist." % dbfile)
                return None
        elif isinstance(dbfile, pd.DataFrame):
            db = dbfile
        self.db = pd.concat(
            [self.db, db], join="outer", ignore_index=True, sort=False, copy=False
        )

        if len(self.db[self.db["use"] == False]) and check_duplicates:
            self.discard_entry(
                self.db[self.db["use"] == False].index.values, save=False
            )

        self.db.drop_duplicates(inplace=True)
        self.db.reset_index(drop=True, inplace=True)

    def get_item(self, item):
        if type(item) == str:
            return pickle.load(open(item, "rb"))
        else:
            return pickle.load(open(self.db.loc[item]["savfile"], "rb"))

    def export_txt(self, name, item, key):
        """Export array to txt file.

        name: filename of exported
        item: db entry to export
        key: key of loaded dictionary to save
        """

        data = self.get_item(item)[key]
        np.savetxt(name, data)
        print("Data saved as ", name)

    @Decorators.input2list
    def rm_db_entry(self, db_id, rmfile=False):
        for i in db_id:
            if rmfile:
                try:
                    os.remove(self.db.loc[i]["savfile"])
                except FileNotFoundError:
                    print("File not found.")
            self.db = self.db.drop(i)
        self.save_db()
