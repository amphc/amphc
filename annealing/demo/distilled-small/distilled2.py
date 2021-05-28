import ctypes
import numpy as np
import numpy
from typing import Any
import time

def gen_timeIndices(sampleRate: float, pulseWidth: float) -> numpy.ndarray[numpy.float64][10000]:
    step = 1.0 / sampleRate
    timeIndices = np.arange(
        -pulseWidth / 2.0, pulseWidth / 2.0 - (1.0 / sampleRate) + step, step
    )
    return timeIndices

def gen_numSamples(timeIndices: numpy.ndarray[numpy.float64][10000]) -> int:
    numSamples = len(timeIndices)
    return numSamples

def gen_numChannels(numPosx: int, numPosy: int) -> int:
    numChannels = numPosx * numPosy
    return numChannels

def gen_SteerVector(
    lightSpeed: float,
    centerFreq: float,
    numPosx: int,
    numPosy: int,
    uAngleArrival: float,
    vAngleArrival: float
) -> numpy.ndarray[numpy.complex128][315]:
    wavelength = lightSpeed / centerFreq
    arrayPosx = np.linspace(-1.5, 1.5, num=numPosx, endpoint=True)
    arrayPosy = np.linspace(-1.05, 1.05, num=numPosy, endpoint=True)
    arrayGridx = np.tile(arrayPosx, (15, 1)).T
    arrayGridy = np.tile(arrayPosy, (21, 1))
    arrayx = np.ravel(arrayGridx.T)
    arrayy = np.ravel(arrayGridy.T)
    steerVector = np.exp(
        1j
        * (2.0 * np.pi / wavelength)
        * (arrayx * uAngleArrival + arrayy * vAngleArrival),
        dtype=np.complex128,
    )
    return steerVector

def gen_fftSize(numSamples: int) -> int:
    fftSize = int(2 * (2 ** (np.ceil(np.log(numSamples) / np.log(2)))))
    return fftSize

def gen_noiseReal(numSamples: int, numChannels: int) -> numpy.ndarray[numpy.float64][315, 10000]:
    noiseReal = np.random.randn(numChannels, numSamples)
    return noiseReal

def gen_noiseImage(numSamples: int, numChannels: int) -> numpy.ndarray[numpy.float64][315, 10000]:
    noiseImag = np.random.randn(numChannels, numSamples)
    return noiseImag

def gen_steerVector1(steerVector: numpy.ndarray[numpy.complex128][315], numChannels: int) -> numpy.ndarray[numpy.complex128][315, 1]:
    steerVector1 = np.reshape(steerVector, (numChannels, 1))
    return steerVector1

def gen_steerVector11(steerVector: numpy.ndarray[numpy.complex128][315], numChannels: int) -> numpy.ndarray[numpy.complex128][1, 315]:
    steerVector11 = np.reshape(steerVector, (1, numChannels))
    steerVector11 = np.conj(steerVector11)
    return steerVector11

def gen_pulseDataNoPadding(
    targetRange0: float,
    targetVelocity: float,
    pulseWidth: float,
    numSamples: int,
    numPulses: int,
    lightSpeed: float,
    sweptWidth: float,
    timeIndices: numpy.ndarray[numpy.float64][10000]
) -> numpy.ndarray[numpy.complex128][32, 10000]:
    pri = pulseWidth / 0.1
    targetRange = targetRange0 + pri * np.arange(numPulses) * targetVelocity
    targetT = 2.0 * targetRange / lightSpeed
    pulseDataNoPadding = np.zeros((numPulses, numSamples), dtype=np.complex128)
    const_0 = np.pi * sweptWidth / pulseWidth  # new variable
    for idx in range(numPulses):
        pulseDataNoPadding[idx, :] = np.exp(
            1j * const_0 * (timeIndices - targetT[idx]) ** 2, dtype=np.complex128
        )
    return pulseDataNoPadding

def gen_d_matchFilter(
    timeIndices: numpy.ndarray[numpy.float64][10000],
    weightFunction: None,
    pulseWidth: float,
    sweptWidth: float,
    fftSize: int
) -> numpy.ndarray[numpy.complex128][32768]:
    x0 = np.exp(
        1j * np.pi * sweptWidth / pulseWidth * (timeIndices ** 2), dtype=np.complex128
    )
    d_out = np.fft.fft(x0, fftSize)
    d_matchFilter = np.conj(d_out)
    d_matchFilterShift = np.fft.fftshift(d_matchFilter)
    if weightFunction is None:
        d_weightFunction = np.ones(fftSize)
    else:
        d_weightFunction = np.asarray(weightFunction)
    d_matchFilterShift = d_matchFilterShift * d_weightFunction
    d_matchFilter = np.fft.fftshift(d_matchFilterShift)
    return d_matchFilter

def gen_beamforming_Z(
    numSamples: int,
    numChannels: int,
    steerVector1: numpy.ndarray[numpy.complex128][315, 1],
    steerVector11: numpy.ndarray[numpy.complex128][1, 315],
    pulseDataNoPadding: numpy.ndarray[numpy.complex128][32, 10000],
    numPulses: int,
    d_matchFilter: numpy.ndarray[numpy.complex128][32768],
    fftSize: int
) -> numpy.ndarray[numpy.complex128][32, 10000]:
    dataCube = np.zeros((numChannels, numSamples), dtype=np.complex128)
    beamforming = np.zeros((numPulses, numSamples), dtype=np.complex128)
    for idx in range(numPulses):
        pulseDataNoPadding1 = np.reshape(pulseDataNoPadding[idx, :], (1, numSamples))
        dataCube = np.multiply(steerVector1, pulseDataNoPadding1, dtype=np.complex128)
        noiseReal = gen_noiseReal(numSamples, numChannels)
        noiseImag = gen_noiseImage(numSamples, numChannels)
        noise = (noiseReal + 1j * noiseImag) / np.sqrt(2.0)
        dataCube = dataCube + noise
        beamforming[idx, :] = np.squeeze(np.matmul(steerVector11, dataCube))
    d_X = np.fft.fft(beamforming, fftSize, axis=1)
    d_matchFilter1 = np.reshape(d_matchFilter, (1, fftSize))
    d_matchFilterMultiply = np.tile(d_matchFilter1, (numPulses, 1))
    d_Y = d_X * d_matchFilterMultiply
    d_y = np.fft.ifft(d_Y, axis=1)
    d_yNorm = d_y / numSamples
    d_yNorm = np.fft.fftshift(d_yNorm, axes=1)
    d_ZTemp = np.fft.fft(d_yNorm, 4 * numPulses, axis=0)
    d_Z = np.fft.fftshift(d_ZTemp, axes=0)
    Z = d_Z  # or np.asnumpy(d_Z)
    return Z

t1 = time.time()
lightSpeed: float = 0.299792458 * 1e9
centerFreq: float = 9e9
sampleRate: float = 1000e6
pulseWidth: float = 10e-6
sweptWidth: float = 100e6
numPulses: int = 32
numPosx: int = 21
numPosy: int = 15

targetVelocity: float = 30.0
targetRange0: float = -250.0
uAngleArrival: float = 0.2
vAngleArrival: float = -0.3

weightFunction: None = None

timeIndices: numpy.ndarray[numpy.float64][10000] = gen_timeIndices(sampleRate, pulseWidth)

numSamples: int = gen_numSamples(timeIndices)
numChannels: int = gen_numChannels(numPosx, numPosy)
steerVector: numpy.ndarray[numpy.complex128][315] = gen_SteerVector(
    lightSpeed, centerFreq, numPosx, numPosy, uAngleArrival, vAngleArrival
)

fftSize: int = gen_fftSize(numSamples)
steerVector1: numpy.ndarray[numpy.complex128][315, 1] = gen_steerVector1(steerVector, numChannels)
steerVector11: numpy.ndarray[numpy.complex128][1, 315] = gen_steerVector11(steerVector, numChannels)
pulseDataNoPadding: numpy.ndarray[numpy.complex128][32, 10000] = gen_pulseDataNoPadding(
    targetRange0,
    targetVelocity,
    pulseWidth,
    numSamples,
    numPulses,
    lightSpeed,
    sweptWidth,
    timeIndices,
)

d_matchFilter: numpy.ndarray[numpy.complex128][32768] = gen_d_matchFilter(
    timeIndices, weightFunction, pulseWidth, sweptWidth, fftSize
)

Z: numpy.ndarray[numpy.complex128][128, 32768] = gen_beamforming_Z(
    numSamples,
    numChannels,
    steerVector1,
    steerVector11,
    pulseDataNoPadding,
    numPulses,
    d_matchFilter,
    fftSize)
t2 = time.time()
print('Time in second: %.6f' %(t2 - t1))
