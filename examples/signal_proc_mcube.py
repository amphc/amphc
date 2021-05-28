import cupy as cp
import numpy as np
import sys

totalcubes = int(sys.argv[1]) if len(sys.argv) > 1 else 2

weightFunction = None
lightSpeed = 0.299792458*1e9  
centerFreq = 9e9              
sampleRate = 1000e6           
pulseWidth = 10e-6            
sweptWidth = 100e6            

pri = pulseWidth / 0.1
numPulses = 32                

targetVelocity = 30.0         
targetRange0 = -250.0         
uAngleArrival = 0.2           
vAngleArrival = -0.3
targetRange = targetRange0 + pri * cp.arange(numPulses) * targetVelocity
targetT = 2.0 * targetRange / lightSpeed

wavelength = lightSpeed / centerFreq
numPosx = 21
numPosy = 15
arrayPosx = cp.linspace(-1.5, 1.5, num=numPosx, endpoint=True)
arrayPosy = cp.linspace(-1.05, 1.05, num=numPosy, endpoint=True)
arrayGridx = cp.tile(arrayPosx, (15,1)).T
arrayGridy = cp.tile(arrayPosy, (21,1))
numChannels = numPosx * numPosy

step = 1.0/sampleRate
timeIndices = cp.arange(-pulseWidth/2.0, pulseWidth/2.0-(1.0/sampleRate)+step, step)
numSamples = len(timeIndices)

pulseDataNoPadding = cp.zeros((numPulses, numSamples), dtype=np.complex128)
for idx in range(numPulses):
    pulseDataNoPadding[idx,:] = cp.exp(1j * cp.pi * sweptWidth / pulseWidth * (timeIndices-targetT[idx])**2, dtype=np.complex128)
    
arrayx = cp.ravel(arrayGridx.T) 
arrayy = cp.ravel(arrayGridy.T)
steerVector = cp.exp(1j * (2.0*cp.pi/wavelength) * (arrayx * uAngleArrival + arrayy * vAngleArrival), dtype=np.complex128)

steerVector1 = cp.reshape(steerVector, (numChannels,1))

steerVector11_0 = cp.reshape(steerVector, (1,numChannels))
steerVector11_1 = cp.conj(steerVector11_0)   

fftSize = int(2 * (2**(cp.ceil(cp.log(numSamples)/cp.log(2)))))

x0 = cp.exp(1j * cp.pi * sweptWidth / pulseWidth * (timeIndices**2), dtype=np.complex128)

d_out = cp.fft.fft(x0, fftSize)
d_matchFilter_0 = cp.conj(d_out)
d_matchFilterShift_0 = cp.fft.fftshift(d_matchFilter_0)

if weightFunction is None:
    d_weightFunction = cp.ones(fftSize)
else:
    d_weightFunction = cp.asarray(weightFunction)

d_matchFilterShift_1 = d_matchFilterShift_0 * d_weightFunction

d_matchFilter_1 = cp.fft.fftshift(d_matchFilterShift_1)

Zs = np.zeros((totalcubes, 4*numPulses, fftSize), dtype=np.complex128)
for cubeid in range(totalcubes):
    dataCube = cp.zeros((numChannels, numSamples), dtype=np.complex128)
    beamforming = cp.zeros((numPulses, numSamples), dtype=np.complex128)
    for idx in range(numPulses):
        pulseDataNoPadding1 = cp.reshape(pulseDataNoPadding[idx,:], (1,numSamples))
        dataCube = cp.multiply(steerVector1, pulseDataNoPadding1, dtype=np.complex128)
        noiseReal = cp.random.randn(numChannels, numSamples)
        noiseImag = cp.random.randn(numChannels, numSamples)
        noise = (noiseReal + 1j*noiseImag) / cp.sqrt(2.0)
        dataCube = dataCube + noise
        beamforming[idx,:] = cp.squeeze(cp.matmul(steerVector11_1, dataCube)) 
    d_X = cp.fft.fft(beamforming, fftSize, axis=1)
    d_matchFilter1 = cp.reshape(d_matchFilter_1, (1,fftSize))
    d_matchFilterMultiply = cp.tile(d_matchFilter1, (numPulses,1))
    d_Y = d_X * d_matchFilterMultiply
    d_y = cp.fft.ifft(d_Y, axis=1)
    d_yNorm_0 = d_y / numSamples
    d_yNorm_1 = cp.fft.fftshift(d_yNorm_0, axes=1)
    d_ZTemp = cp.fft.fft(d_yNorm_1, 4*numPulses, axis=0)
    d_Z = cp.fft.fftshift(d_ZTemp, axes=0)
    Z = cp.asnumpy(d_Z)
    Zs[cubeid,:,:] = Z
