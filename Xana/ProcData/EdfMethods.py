from os.path import isfile
from .EdfFile3 import EdfFile, EdfGzipFile

####################################
# --- Standard EDF w/r functions ---#
####################################


def loadedf(filename, imgn=0):
    if isfile(filename):
        if filename.endswith("edf"):
            f = EdfFile(filename)
        elif filename.endswith("edf.gz"):
            f = EdfGzipFile(filename)
        return f.GetData(imgn)
    else:
        print("file ", filename, " does not exist!")
        return 0


def saveedf(filename, data, imgn=0):
    try:
        newf = EdfFile(filename)
        newf.WriteImage({}, data, imgn)
        print("file is saved to ", filename)
        return
    except:
        print("file is not saved!")
        return


def headeredf(filename, imgn=0):
    if isfile(filename):
        if filename.endswith("edf"):
            f = EdfFile(filename)
        elif filename.endswith("edf.gz"):
            f = EdfGzipFile(filename)
        return f.GetHeader(imgn)
    else:
        print("file ", filename, " does not exist!")
        return 0
