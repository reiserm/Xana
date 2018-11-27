import numpy as np
import time
import multiprocessing as mp
from multiprocessing.managers import SyncManager
from queue import PriorityQueue
from XpcsAna.Xpcs import Xpcs
from XsvsAna.Xsvs import Xsvs
from SaxsAna.Saxs import Saxs
from ProcData.Xdata import Xdata
from Setup import Setup
from Xdb import Xdb
from Decorators import Decorators
from misc.xsave import mksavdir, save_result, make_filename
from helper import *
import copy


class MyManager(SyncManager):
    pass


def Manager():
    m = MyManager()
    m.start()
    return m


class Analysis(Xdata):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    @Decorators.input2list
    def analyze(self, series_id, method, first=0, last=np.inf, handle_existing='next',
                nread_procs=4, chunk_size=200, verbose=True, dark=None,
                dtype=np.float32, filename='', read_kwargs={}, **kwargs):

        for sid in series_id:
            if verbose:
                print('\n\n#### Starting %s Analysis ####\nSeries: %d in folder %s\n' %
                      (method, sid, self.datdir))
                print('Using {} processes to read data.'.format(nread_procs))

            # if dark is not None:
            #     if type(dark) == int:
            #         print('Loading DB entry {} as dark.'.format(dark))
            #         dark = self.xana.get_item(dark)['Isaxs']

            last = min([self.meta.loc[sid, 'nframes'], last])
            
            read_opt = {'first': (first,),
                        'last': (last,),
                        'dark': dark,
                        'verbose': False,
                        'dtype': dtype,
                        'qsec': self.setup['qsec'],
                        'output': '2dsection',
                        'nprocs': nread_procs,
                        'chunk_size':chunk_size
                        }
            saxs_dict = read_opt.copy()
            read_opt.update(read_kwargs)

            qsec = self.setup['qsec']
            proc_dat = {'nimages': last - first,
                        'dim': (qsec[1][0] - qsec[0][0] + 1, qsec[1][1] - qsec[0][1] + 1)
                        }

            fmax = self.meta.loc[sid, 'nframes']
            chunks = [np.arange(first + i*chunk_size,
                                min([min(first + (i + 1)*chunk_size, last), fmax]))
                      for i in range(np.ceil((last - first) / chunk_size).astype(np.int32))]

            if method in ['xpcs', 'xsvs']:

                # Register a shared PriorityQueue
                MyManager.register("PriorityQueue", PriorityQueue)
                m = Manager()
                dataQ = m.PriorityQueue(nread_procs)
                indxQ = m.PriorityQueue()
                #dataQ = mp.Queue(nread_procs)
                #indxQ = mp.Queue()

                # add queues to read and process dictionaries
                read_opt['dataQ'] = dataQ
                read_opt['indxQ'] = indxQ
                read_opt['method'] = 'queue_chunk'
                proc_dat['dataQ'] = dataQ

                for i, chunk in enumerate(chunks):
                    indxQ.put((i, chunk))

                # h5 files can only be opened by one process at a time and, therefore,
                # the processes have to acquire a lock for reading data
                lock = 0
                if 'h5' in self.fmtstr:
                    lock = mp.Lock()
                    read_opt['lock'] = lock

                procs = []
                for ip in range(nread_procs):
                    procs.append(mp.Process(target=self.get_series,
                                            args=(sid,), kwargs=read_opt))
                    procs[ip].start()
                    time.sleep(2)

            if method == 'xpcs':
                saxs = kwargs.pop('saxs', 'compute')
                Isaxs = self.get_xpcs_args(sid, saxs, saxs_dict)
                dt = self.get_delay_time(sid)
                savd = Xpcs.pyxpcs(proc_dat, self.setup['qroi'], dt=dt, qv=self.setup['qv'],
                                   saxs=Isaxs, mask=self.mask, ctr=self.setup['ctr'],
                                   qsec=self.setup['qsec'][0], **kwargs)

            elif method == 'xpcs_evt':
                dt = self.get_delay_time(sid)
                evt_dict = dict(method='events',
                                verbose=True,
                                qroi=self.setup['qroi'],
                                dtype=np.uint32,
                )
                read_opt.update(evt_dict)
                evt = self.get_series(sid, **read_opt)
                savd = Xpcs.eventcorrelator(evt[1:], self.setup['qroi'], self.setup['qv'],
                                            dt, method='events', **kwargs)

            elif method == 'xsvs':

                t_e = self.get_xsvs_args(sid,)
                savd = Xsvs.pyxsvs(proc_dat, self.setup['qroi'], t_e=t_e,
                                   qv=self.setup['qv'], qsec=self.setup['qsec'][0],
                                   **kwargs)

            elif method == 'saxs':

                read_opt['output'] = '2d'
                proc_dat = {'get_series': self.get_series,
                            'sid': sid,
                            'setup': self.setup,
                            'mask': self.mask}
                savd = Saxs.pysaxs(proc_dat, **read_opt, **kwargs)

            else:
                raise ValueError('Analysis type %s not understood.' % method)

            if method in ['xpcs', 'xsvs']:
                # stopping processes
                for ip in range(nread_procs):
                    procs[ip].join()
                # closing queues
                # dataQ.close()
                # dataQ.join_thread()
                # indxQ.close()
                # indxQ.join_thread()

            f = self.datdir.split('/')[-2] + '_s' + \
                str(self.meta.loc[sid, 'series']) + filename
            savfile = save_result(
                savd, method, self.savdir, f, handle_existing)

            self.add_db_entry(sid, savfile, method)

    def get_xpcs_args(self, sid, saxs, read_opt):
        ''' Get Saxs and delay time for XPCS analysis.
        '''
        if saxs == 'compute':
            print('Calculating average SAXS image.')
            Isaxs = self.get_series(sid, method='average', **read_opt)[0]
        elif isinstance(saxs, int):
            if saxs == -1:
                saxs = self.db.shape[0] - 1
            print('Loading average SAXS from database entry {}'.format(saxs))
            Isaxs = self.get_item(saxs)['Isaxs']
        else:
            Isaxs = saxs
        return Isaxs

    def get_delay_time(self, sid):
        dt = 0
        for attr in ['t_delay', 't_exposure', 't_readout', 't_latency', 'rate',
                     'pulseLength']:
            if attr in self.meta.columns:
                item = self.meta.loc[sid, attr]
                if attr == 'rate':
                    dt += 1/item
                elif attr == 'pulseLength':
                    dt += item * 1e-15
                else:
                    dt += item
                    if attr == 't_delay':
                        break
        return dt

    def get_xsvs_args(self, sid):
        ''' Get exposure time for XSVS analysis
        '''
        t_e = 0
        for attr in ['t_exposure', 'pulseLength']:
            if attr in self.meta.columns:
                item = self.meta.loc[sid, attr]
                if attr == 'pulseLength':
                    t_e += item * 1e15
                else:
                    t_e += item

        return t_e

    def defineqrois(self, input_, **kwargs):
        if type(input_) == int:
            Isaxs = self.get_item(input_)['Isaxs']
        elif type(input_) == np.ndarray:
            Isaxs = input_
        if Isaxs.ndim == 3:
            Isaxs = self.arrange_tiles(Isaxs)
        Saxs.defineqrois(self.setup, Isaxs, self.mask, **kwargs)

    @staticmethod
    def find_center(*args, **kwargs):
        return Saxs.find_center(*args, **kwargs)
