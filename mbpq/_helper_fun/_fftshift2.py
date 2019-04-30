import numpy as np


def fftshift2(I):
    if I.ndim >= 3:
        S = np.fft.fftshift(np.fft.fftshift(I, -2), -1)
    else:
        S = np.fft.fftshift(np.fft.fftshift(I, 1), 0)
    return S
