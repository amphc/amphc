import numpy as np

import numpy
from typing import Any
np: module
def gen_timeIndices(sampleRate: float, pulseWidth: float) -> numpy.ndarray[numpy.int64][10000]:
    step = 1.0/sampleRate
    timeIndices = np.arange(-pulseWidth/2.0, pulseWidth/2.0-(1.0/sampleRate)+step, step)
    return timeIndices

def gen_steerVector(lightSpeed: float, centerFreq: float, uAngleArrival: float, vAngleArrival: float, numPosx: int, numPosy: int) -> numpy.ndarray[numpy.complex128][315]:
    arrayPosx = np.linspace(-1.5, 1.5, num=numPosx, endpoint=True)
    arrayPosy = np.linspace(-1.05, 1.05, num=numPosy, endpoint=True)
    arrayGridx = np.tile(arrayPosx, (15,1)).T
    arrayGridy = np.tile(arrayPosy, (21,1))
    wavelength = lightSpeed / centerFreq
    arrayx = np.ravel(arrayGridx.T) 
    arrayy = np.ravel(arrayGridy.T)
    steerVector = np.exp(1j * (2.0*np.pi/wavelength) * (arrayx * uAngleArrival + arrayy * vAngleArrival), dtype=np.complex128)
    return steerVector

def gen_numChannels(numPosx: int, numPosy: int) -> int:
    numChannels = numPosx * numPosy
    return numChannels

def gen_numSamples(timeIndices: numpy.ndarray[numpy.int64][10000]) -> int:
    numSamples = len(timeIndices)
    return numSamples

def gen_steerVector1(steerVector: numpy.ndarray[numpy.complex128][315], numChannels: int) -> numpy.ndarray[numpy.complex128][315, 1]:
    steerVector1 = np.reshape(steerVector, (numChannels,1))
    return steerVector1

def gen_steerVector11_1(steerVector: numpy.ndarray[numpy.complex128][315], numChannels: int) -> numpy.ndarray[numpy.complex128][1, 315]:
    steerVector11_0 = np.reshape(steerVector, (1,numChannels))
    steerVector11_1 = np.conj(steerVector11_0)
    return steerVector11_1

def gen_pulseDataNoPadding(lightSpeed: float, pulseWidth: float, targetRange0: float, targetVelocity: float, numPulses: int, sweptWidth: float, timeIndices: numpy.ndarray[numpy.int64][10000], numSamples: int) -> numpy.ndarray[numpy.complex128][32, 10000]:
    pri = pulseWidth / 0.1
    targetRange = targetRange0 + pri * np.arange(numPulses) * targetVelocity
    targetT = 2.0 * targetRange / lightSpeed
    pulseDataNoPadding = np.zeros((numPulses, numSamples), dtype=np.complex128)
    for idx in range(numPulses):
        pulseDataNoPadding[idx,:] = np.exp(1j * np.pi * sweptWidth / pulseWidth * (timeIndices-targetT[idx])**2, dtype=np.complex128)
    return pulseDataNoPadding

def gen_fftSize(numSamples: int) -> int:
    fftSize = int(2 * (2**(np.ceil(np.log(numSamples)/np.log(2)))))
    return fftSize

def gen_d_matchFilter_1(weightFunction: None, pulseWidth: float, sweptWidth: float, timeIndices: numpy.ndarray[numpy.int64][10000], fftSize: int) -> numpy.ndarray[numpy.complex128][32768]:
    x0 = np.exp(1j * np.pi * sweptWidth / pulseWidth * (timeIndices**2), dtype=np.complex128)
    d_out = np.fft.fft(x0, fftSize)
    d_matchFilter_0 = np.conj(d_out)
    d_matchFilterShift_0 = np.fft.fftshift(d_matchFilter_0)
    if weightFunction is None:
        d_weightFunction = np.ones(fftSize)
    else:
        d_weightFunction = np.asarray(weightFunction)
    d_matchFilterShift_1 = d_matchFilterShift_0 * d_weightFunction
    d_matchFilter_1 = np.fft.fftshift(d_matchFilterShift_1)
    return d_matchFilter_1

def gen_Zs(d_matchFilter_1: numpy.ndarray[numpy.complex128][32768], fftSize: int, numPulses: int, numSamples: int, pulseDataNoPadding: numpy.ndarray[numpy.complex128][32, 10000], totalcubes: int) -> numpy.ndarray[numpy.complex128][2, 128, 32768]:
    Zs = np.zeros((totalcubes, 4*numPulses, fftSize), dtype=np.complex128)
    for cubeid in range(totalcubes):
        dataCube = np.zeros((numChannels, numSamples), dtype=np.complex128)
        beamforming = np.zeros((numPulses, numSamples), dtype=np.complex128)
        for idx in range(numPulses):
            pulseDataNoPadding1 = np.reshape(pulseDataNoPadding[idx,:], (1,numSamples))
            dataCube = np.multiply(steerVector1, pulseDataNoPadding1, dtype=np.complex128)
            noiseReal = np.random.randn(numChannels, numSamples)
            noiseImag = np.random.randn(numChannels, numSamples)
            noise = (noiseReal + 1j*noiseImag) / np.sqrt(2.0)
            dataCube = dataCube + noise
            beamforming[idx,:] = np.squeeze(np.matmul(steerVector11_1, dataCube)) 
        d_X = np.fft.fft(beamforming, fftSize, axis=1)
        d_matchFilter1 = np.reshape(d_matchFilter_1, (1,fftSize))
        d_matchFilterMultiply = np.tile(d_matchFilter1, (numPulses,1))
        d_Y = d_X * d_matchFilterMultiply
        d_y = np.fft.ifft(d_Y, axis=1)
        d_yNorm_0 = d_y / numSamples
        d_yNorm_1 = np.fft.fftshift(d_yNorm_0, axes=1)
        d_ZTemp = np.fft.fft(d_yNorm_1, 4*numPulses, axis=0)
        d_Z = np.fft.fftshift(d_ZTemp, axes=0)
        Z = d_Z
        Zs[cubeid,:,:] = Z
    return Zs
    
weightFunction: None = None
lightSpeed: float = 0.299792458*1e9  
centerFreq: float = 9e9              
sampleRate: float = 1000e6           
pulseWidth: float = 10e-6            
sweptWidth: float = 100e6            
numPulses: int = 32                
targetVelocity: float = 30.0         
targetRange0: float = -250.0         
uAngleArrival: float = 0.2           
vAngleArrival: float = -0.3
numPosx: int = 21
numPosy: int = 15
totalcubes: int = 2

timeIndices: numpy.ndarray[numpy.int64][10000] = gen_timeIndices(sampleRate, pulseWidth)
steerVector: numpy.ndarray[numpy.complex128][315] = gen_steerVector(lightSpeed, centerFreq, uAngleArrival, vAngleArrival, numPosx, numPosy)
numChannels: int = gen_numChannels(numPosx, numPosy)

numSamples: int = gen_numSamples(timeIndices)
steerVector1: numpy.ndarray[numpy.complex128][315, 1] = gen_steerVector1(steerVector, numChannels)
steerVector11_1: numpy.ndarray[numpy.complex128][1, 315] = gen_steerVector11_1(steerVector, numChannels)

pulseDataNoPadding: numpy.ndarray[numpy.complex128][32, 10000] = gen_pulseDataNoPadding(lightSpeed, pulseWidth, targetRange0, targetVelocity, numPulses, sweptWidth, timeIndices, numSamples)
fftSize: int = gen_fftSize(numSamples)

d_matchFilter_1: numpy.ndarray[numpy.complex128][32768] = gen_d_matchFilter_1(weightFunction, pulseWidth, sweptWidth, timeIndices, fftSize)

Zs: numpy.ndarray[numpy.complex128][2, 128, 32768] = gen_Zs(d_matchFilter_1, fftSize, numPulses, numSamples, pulseDataNoPadding, totalcubes)
