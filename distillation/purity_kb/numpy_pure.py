import numpy as np
from numpy import random
from numpy import fft

numpy_pure_funcs = [
	np.arange, np.linspace, np.tile, np.ravel, np.exp, np.ceil, np.sqrt,
	np.log, np.reshape, np.conj, np.zeros, np.ones, np.asarray,
	np.multiply, np.squeeze, np.matmul,
	np.fft.fft, np.fft.fftshift, np.fft.ifft,
	np.random.randn
]