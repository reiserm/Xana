# Miscellaneous functions used in different modules


def attrstate(d, attr):

    if d:
        if attr in d:
            if d[attr]:
                return True
    else:
        return False
