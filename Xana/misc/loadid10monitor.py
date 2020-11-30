import numpy as np
import pandas as pd
import re
from io import StringIO


def loadmonitor(f, scan_time="3000-01-01 00:00:00", scanid=-1):
    """Load ID10 meta information from monitor file f.
    The time stamp of the measurement is required.
    """
    scan_time = pd.to_datetime(scan_time)
    dummy_time = 0
    i = 0
    with open(f, "r") as monitor:
        line = monitor.readline()
        while line:
            if line.startswith("#S"):
                S = re.search("(?<=S )\d*", line).group(0)
                date = monitor.readline()
                line += date
                date = date.lstrip("#D ")
                nearest_time = pd.to_datetime(date)
                if nearest_time > scan_time or int(S) == scanid:
                    scan = ""
                    while line:
                        scan += line
                        line = monitor.readline()
                        if len(line.strip()) == 0:
                            print("Scan ID is ", S)
                            print("Closest time is ", nearest_time)
                            break
                    break
            else:
                pass
            line = monitor.readline()

    hline, lline = np.where([0 if x.startswith("#") else 1 for x in scan.split("\n")])[
        0
    ][[0, -2]]
    header = re.split(r"\s{2,}", scan.split("\n")[hline - 1])
    header[0] = header[0][3:]
    df = pd.read_csv(
        StringIO(scan),
        names=header,
        skiprows=hline,
        nrows=lline - hline + 1,
        delim_whitespace=1,
    )

    return df, scan, (hline - 1, lline + 1)
