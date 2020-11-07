import numpy as np
import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack as fp


def cross_correlation(q):
    ind = np.argmin(np.abs(q - 0.6))
    inten = I2d[:, ind]
    print("Found %d angles" % len(inten))
    phi = chi.copy()

    N = len(inten)
    lags = np.arange(-N + 1, N)

    inten = inten - inten.mean()

    acf = np.correlate(inten, inten, mode="full")
    acf /= N * inten.std() ** 2

    l = len(acf)
    x = np.linspace(-4 * np.pi, 4 * np.pi, l)
    print(N, "Mean Intensity is: " + str(mean_int_2))
    figure()
    plt.plot(lags, acf, "-")
    plt.grid()

    fourier = fp.fft(acf)
    xf = fp.fftfreq(l, d=1 / (2 * 360))
    xf = fp.fftshift(xf)
    fourier = fp.fftshift(fourier)
    fourier = 1.0 / l * np.absolute(fourier)
    # fourier = fourier[1:]
    # xf = xf[:l//2]

    plt.figure()
    plt.plot(xf, fourier)
    plt.grid()
    # # plt.savefig(savdir + 'phi_corr_long_fourier.png', format='png',dpi=400)

    # plt.show()
