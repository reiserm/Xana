# cython: boundscheck=False, wraparound=False, cdivision=True

from numpy cimport ndarray

def fecorrt3m(ndarray[int, ndim=1, mode='fortran'] pix,
              ndarray[int, ndim=1, mode='fortran'] t,
              ndarray[int, ndim=2, mode='fortran'] cc,
              int lpi, int nt):

    cdef int t0, i = 0, j = 0

    for i in range(lpi):
        t0 = t[i]
        j = i + 1

        while pix[j] == pix[i]:
            cc[t[j], t0] = cc[t[j], t0] + 1
            j += 1

            if j == lpi:
                break

        i += 1

    # Common math operation of some name.
    for i in range(nt - 1):
        for j in range(nt - 1):
            cc[j, i] = cc[j, i] + cc[i, j]
            cc[i, j] = cc[j, i]
