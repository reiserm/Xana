import readline


def histsave(f, histfile="./.save_history"):
    readline.clear_history()
    try:
        readline.read_history_file(histfile)
        h_len = readline.get_current_history_length()
    except FileNotFoundError:
        open(histfile, "wb").close()
        h_len = 0

    readline.add_history(f)
    readline.write_history_file(histfile)


def get_history_list(histfile="~/xana_save_history", ftype="setup"):
    histlist = []
    with open(histfile, "r") as f:
        line = f.readline()
        while line:
            if ftype in line.split("/")[-1]:
                histlist.append(line.rstrip("\n"))
            line = f.readline()
    return np.unique(histlist)
