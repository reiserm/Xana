from pathlib import Path
import os
import re
import glob
import pickle


def make_filename(obj, attr=None, filename=None):

    if attr is None and filename is None:
        raise ValueError(
            "Both attribute name and filename are None." "Cannot create filename."
        )

    p = obj.savdir
    if attr is not None:
        orgname = getattr(obj, attr)
    if filename is None:
        p = Path(orgname)
    else:
        p = Path(filename).resolve()
    return p


def mksavdir(sample_name=None, savhome="./", handle_existing="use"):
    """Create a directory for saving results."""

    savhome = Path(savhome).resolve()
    if sample_name is None:
        sample_name = input(
            ("Chose a sample name for saving in {}\n" + "or enter full path.").format(
                savhome
            )
        )
    savdir = savhome.joinpath(sample_name)

    try:
        savdir.mkdir()
    except FileExistsError:
        if handle_existing == "use":
            pass
        elif handle_existing == "next":
            searchstr = r"_\d{2}$"
            savsplit = savdir.split("/")
            reg = re.search(searchstr, savsplit[-2])
            folderlist = glob.glob(savdir[:-1] + "_*")
            if reg is None and len(folderlist) == 0:
                savsplit[-2] += "_02"
            else:
                counter = max(
                    list(
                        map(
                            lambda x: int(re.search(searchstr, x).group().lstrip("_")),
                            folderlist,
                        )
                    )
                )
                savsplit[-2] += "_{:02d}".format(counter + 1)
            savdir = "/".join(savsplit)
            os.makedirs(savdir)
        else:
            print("Could not create directory.")
    savdir = os.path.abspath(savdir) + "/"
    print("Changing savdir to:\n\t{}".format(savdir))
    return savdir


def save_result(
    savobj, restype, savdir, filename="", handle_existing="raise", prompt=True
):

    if savdir is None:
        savdir = mksavdir("")
    savdir = Path(savdir).resolve()
    # if not str(filename).startswith(restype):
    #     filename = f"{restype}_{filename}"
    savname = savdir.joinpath(filename)

    save = True
    if handle_existing == "next":

        def find_counter(x):
            tmp = re.search(r"(?<=_)\d{4}$", x.stem)
            return int(tmp.group()) if bool(tmp) else -1

        filelist = savdir.rglob("*.pkl")
        counter = [find_counter(x) for x in filelist]
        counter = max(counter) + 1
        savname = savdir.joinpath(filename + "_{:04d}".format(counter) + ".pkl")
    elif handle_existing in ["overwrite", "w"]:
        pass
    elif handle_existing == "raise":
        if savname.is_file() and not prompt:
            raise OSError(
                (
                    "File {} already exists. Change overwrite to "
                    + "True or choose different name."
                ).format(savname)
            )
        elif savname.is_file() and prompt:
            user_input = input("File exists. Save anyway? (No/Yes)\t")
            if user_input.lower() == "yes":
                pass
            else:
                save = False
    else:
        raise ValueError("{} is not a valid option".format(handle_existing))

    if save:
        savname = savname.with_suffix(".pkl")
        pickle.dump(savobj, open(savname, "wb"))
        print("\nResults saved to:\n\t{}".format(savname))
        return savname
    else:
        print("Result has not been saved.")
