import cupy as cp
from cupy import random
from cupy import fft

cupy_pure_funcs = [
	cp.arange, cp.linspace, cp.tile, cp.ravel, cp.exp, cp.ceil, cp.sqrt,
	cp.log, cp.reshape, cp.conj, cp.zeros, cp.ones, cp.asarray,
	cp.multiply, cp.squeeze, cp.matmul,
	cp.fft.fft, cp.fft.fftshift, cp.fft.ifft,
	cp.random.randn
]