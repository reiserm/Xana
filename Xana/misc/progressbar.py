def progress(count, total, suffix=""):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = "=" * filled_len + "-" * (bar_len - filled_len)
    if percents < 100:
        print("\r[{}] {}{} ...{}".format(bar, percents, "%", suffix), end="", flush=1)
    else:
        print("\r[{}] {}{}".format(bar, percents, "%"), end="\n", flush=1)
