from ProcData.ReadData import read_data
from ProcData.getmeta import getfiles, files2series, get_attrs_from_dict, get_attrs_h5, get_header_h5
from ProcData.H5Methods import common_mode_from_hist

class Xfmt:
    """Class that defines all methods and attributes to handle
    dfferent data types.
    """

    def __init__(self, fmtstr=''):

        self.fmtstr = fmtstr
        if fmtstr == '':
            print('Format string is empty. Specify valid data format to load data.')
        else:
            if fmtstr == 'id10_eiger_single_edf':
                kernel = self.__init_id10_eiger_single_edf()
            elif fmtstr == 'pilatus_single_cbf':
                kernel = self.__init_pilatus_single_cbf()
            elif fmtstr == 'p10_eiger_h5':
                kernel = self.__init_p10_eiger_h5()
            elif fmtstr == 'xcs_cspad_h5':
                kernel = self.__init_xcs_cspad_h5()
            elif fmtstr == 'converted_h5':
                kernel = self.__init_converted_h5()
            elif fmtstr == 'lambda_nxs':
                kernel = self.__init_lambda_nxs()
            elif fmtstr == 'id02_eiger_single_edf':
                kernel = self.__init_id02_eiger_single_edf()
            elif fmtstr == 'id02_eiger_multi_edf':
                kernel = self.__init_id02_eiger_multi_edf()
            self.__dict__.update(kernel)
            self.load_data_func = read_data
            self.getfiles = getfiles
            self.files2series = files2series

    @staticmethod
    def __init_id10_eiger_single_edf():
        import ProcData.EdfMethods as edf
        kernel = {
            'prefix': 'img_eiger',
            'suffix': '((edf)$|(edf\.gz)$)',
            'numfmt': '\d{4}',
            'masterfmt': '((img_eiger_\d{4}_0000_0000)|(img_0{4})|(zaptime_\d{,5}_eiger1_0{4}))',
            'seriesfmt': '\d{1,5}',
            'get_header': edf.headeredf,
            'get_attributes': get_attrs_from_dict,
            'attributes': {
                't_exposure': ('acq_expo_time', float),
                't_readout': ('ccd_readout_time', float),
                't_latency': ('acq_latency_time', float),
                'nframes': ('acq_nb_frames', int)},
        }
        return kernel

    @staticmethod
    def __init_id02_eiger_single_edf():
        import ProcData.EdfMethods as edf
        kernel = {
            'prefix': '',
            'suffix': '((edf)$|(edf\.gz)$)',
            'numfmt': '((?<=_)\d{4,}(?=\.))',
            'masterfmt': '(.*_\d{5}_00001)',
            'seriesfmt': '\d{5}',
            'get_header': edf.headeredf,
            'get_attributes': get_attrs_from_dict,
            'attributes': {
                't_exposure': ('acq_expo_time', float),
                't_readout': ('ccd_readout_time', float),
                't_latency': ('acq_latency_time', float),
                'nframes': ('acq_nb_frames', int)},
        }
        return kernel

    @staticmethod
    def __init_id02_eiger_multi_edf():
        import ProcData.EdfMethods as edf
        kernel = {
            'prefix': '',
            'suffix': '((edf)$|(edf\.gz)$)',
            'numfmt': '((?<=_)\d{4,}(?=\.))',
            'masterfmt': '(.*_\d{5}_00001)',
            'seriesfmt': '\d{1,5}',
            'get_header': edf.headeredf,
            'get_attributes': get_attrs_from_dict,
            'attributes': {
                't_exposure': ('acq_expo_time', float),
                't_readout': ('ccd_readout_time', float),
                't_latency': ('acq_latency_time', float),
                'nframes': ('acq_nb_frames', int)},
        }
        return kernel

    @staticmethod
    def __init_pilatus_single_cbf():
        import ProcData.CbfMethods as cbf
        kernel = {
            'prefix': 'img_eiger',
            'suffix': '(cbf)$',
            'numfmt': '\d{5}(?=\.)',
            'masterfmt': '.*_\d{5}_00001',
            'seriesfmt': '(?<=_)\d{5}(?=_)',
            'get_header': cbf.headercbf,
            'get_attributes': get_attrs_from_dict,
            'attributes': {
                't_exposure': ('acq_expo_time', float),
                't_readout': ('ccd_readout_time', float),
                't_latency': ('acq_latency_time', float),
                'nframes': ('acq_nb_frames', int)},
        }
        return kernel

    @staticmethod
    def __init_p10_eiger_h5():
        kernel = {
            'prefix': '',
            'suffix': 'h5',
            'numfmt': '\d{5,6}',
            'masterfmt': '.*_master',
            'seriesfmt': '\d{5}((?=_master)|(?=_data))',
            'get_header': get_header_h5,
            'get_attributes': get_attrs_h5,
            'attributes': {
                't_delay': ('/entry/instrument/detector/frame_time', 'float32'),
                't_exposure': ('/entry/instrument/detector/count_time', 'float32'),
                't_readout': ('/entry/instrument/detector/detector_readout_time', 'float32'),
                'nframes': ('/entry/instrument/detector/detectorSpecific/nimages', 'int32')
            },
            'h5opt': {'driver': 'stdio',
                      'data': '/entry/data/',
                      'ExternalLinks': True,
                      'chunk_size': 200,
                      'images_per_file': 2000,
                      },
        }
        return kernel

    @staticmethod
    def __init_xcs_cspad_h5():
        from ProcData.ArrangeModules import arrange_cspad_tiles
        from Xdrop.DropletizeData import dropletizedata
        kernel = {
            'prefix': '',
            'suffix': 'h5',
            'numfmt': '(?<=-r)\d{4,5}',
            'masterfmt': '.*-r\d{4,5}',
            'seriesfmt': '(?<=-r)\d{4,5}',
            'get_header': get_header_h5,
            'get_attributes': get_attrs_h5,
            'attributes': {
                'rate': ('/Configure:0000/Run:0000/CalibCycle:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0/Rate/data', 'float32'),
                'pulseLength': ('/Configure:0000/Run:0000/CalibCycle:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0/Pulse length/data', 'float32'),
                'photonEnergy': ('/Configure:0000/Run:0000/CalibCycle:0000/Epics::EpicsPv/EpicsArch.0:NoDevice.0/Photon beam energy/data', 'float32'),
                'nframes': (False, 'int32')
            },

            'h5opt': {'driver': 'stdio',
                      'data': '/Configure:0000/Run:0000/CalibCycle:0000/CsPad::ElementV2/XcsEndstation.0:Cspad.0/data',
                      'ExternalLinks': False,
                      'chunk_size': 256,
                      'commonmode': common_mode_from_hist,
                      'arrange_tiles': arrange_cspad_tiles,
                      'dropletize': dropletizedata
                      },
        }
        return kernel

    @staticmethod
    def __init_converted_h5():
        kernel = {
            'prefix': '',
            'suffix': 'h5',
            'numfmt': '(((_\d{4}){2,}(?=\.)))',  # '|(_\d{4,}(?=\.)))',
            'masterfmt': '((img_eiger_\d{4}_0000_0000)|(img_0{4})|(zaptime_\d{,5}_eiger1_0{4}))',
            'seriesfmt': '\d{1,5}',
            'get_header': get_header_h5,
            'get_attributes': get_attrs_h5,
            'attributes': {
                't_exposure': ('/data/', 'float32'),
                't_readout': ('/data/', 'float32'),
                't_latency': ('/data/', 'float32'),
                'nframes': ('/data/', 'int32')
            },
            'h5opt': {'driver': 'stdio',
                      'data': '/data/',
                      'ExternalLinks': False,
                      'chunk_size': 200,
                      'images_per_file': 1e6,
                      },
        }
        return kernel

    @staticmethod
    def __init_lambda_nxs():
        kernel = {
            'prefix': '',
            'suffix': 'nxs',
            'numfmt': '(?<=_)\d{5}(?=-)',
            'masterfmt': '.*_\d{5,}-\d{5,}',
            'seriesfmt': '(?<=_)\d{5,}(?=_)',
            'get_header': get_header_h5,
            'get_attributes': get_attrs_h5,
            'attributes': {
                't_exposure': ('/entry/instrument/detector/count_time', 'float32'),
                't_readout': ('/entry/instrument/detector/detector_readout_time', 'float32'),
                't_latency': ('/entry/instrument/detector/collection/delay_time', 'float32'),
                'nframes': ('/entry/instrument/detector/collection/frame_numbers', 'int32')
            },
            'h5opt': {'driver': 'stdio',
                      'data': '/entry/instrument/detector/data',
                      'ExternalLinks': False,
                      'chunk_size': 200,
                      'images_per_file': 1e6,
                      },
        }
        return kernel
