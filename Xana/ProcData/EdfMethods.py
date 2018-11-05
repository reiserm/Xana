from os.path import isfile
from ProcData import EdfFile3

####################################
#--- Standard EDF w/r functions ---#
####################################


def loadedf(filename, imgn=0):
    if isfile(filename):
        if filename.endswith('edf'):
            f = EdfFile3.EdfFile(filename)
        elif filename.endswith('edf.gz'):
            f = EdfFile3.EdfGzipFile(filename)
        return f.GetData(imgn)
    else:
        print("file ", filename, " does not exist!")
        return 0


def saveedf(filename, data, imgn=0):
    try:
        newf = EdfFile3.EdfFile(filename)
        newf.WriteImage({}, data, imgn)
        print("file is saved to ", filename)
        return
    except:
        print("file is not saved!")
        return


def headeredf(filename, imgn=0):
    if isfile(filename):
        if filename.endswith('edf'):
            f = EdfFile3.EdfFile(filename)
        elif filename.endswith('edf.gz'):
            f = EdfFile3.EdfGzipFile(filename)
        return f.GetHeader(imgn)
    else:
        print("file ", filename, " does not exist!")
        return 0
