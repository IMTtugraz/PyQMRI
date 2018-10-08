import numpy as np

def  fftshift3(I):
  S = np.fft.fftshift(np.fft.fftshift(np.fft.fftshift(I,-1),-2),-3)
  return S

