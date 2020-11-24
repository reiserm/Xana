from os.path import isfile

# import cbf

####################################
# --- Standard cbf w/r functions ---#
####################################


def loadcbf(filename):
    if isfile(filename):
        f = cbf.read(filename)
        return f.data
    else:
        print("file ", filename, " does not exist!")
        return 0


def savecbf(filename, data):
    if not isfile(filename):
        cbf.write(filename, data)
    else:
        print("file ", filename, " does already exist!")
    return None


def headercbf(filename):
    if isfile(filename):
        f = cbf.read(filename)
        return f.metadata
    else:
        print("file ", filename, " does not exist!")
        return 0
