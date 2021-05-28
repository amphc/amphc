import numpy as np
import sys

totalcubes = int(sys.argv[1]) if len(sys.argv) > 1 else 1

weightFunction = None
lightSpeed = 0.299792458*1e9  
centerFreq = 9e9              
sampleRate = 1000e6           
pulseWidth = 30e-6            
sweptWidth = 100e6            

pri = pulseWidth / 0.1
numPulses = 100

targetVelocity = 30.0         
targetRange0 = -250.0         
uAngleArrival = 0.2           
vAngleArrival = -0.3
targetRange = targetRange0 + pri * np.arange(numPulses) * targetVelocity
targetT = 2.0 * targetRange / lightSpeed

wavelength = lightSpeed / centerFreq
numPosx = 40
numPosy = 25
arrayPosx = np.linspace(-1.5, 1.5, num=numPosx, endpoint=True)
arrayPosy = np.linspace(-1.05, 1.05, num=numPosy, endpoint=True)
arrayGridx = np.tile(arrayPosx, (numPosy,1)).T
arrayGridy = np.tile(arrayPosy, (numPosx,1))
numChannels = numPosx * numPosy

step = 1.0/sampleRate
timeIndices = np.arange(-pulseWidth/2.0, pulseWidth/2.0-(1.0/sampleRate)+step, step)
numSamples = len(timeIndices)

pulseDataNoPadding = np.zeros((numPulses, numSamples), dtype=np.complex128)
for idx in range(numPulses):
    pulseDataNoPadding[idx,:] = np.exp(1j * np.pi * sweptWidth / pulseWidth * (timeIndices-targetT[idx])**2, dtype=np.complex128)
    
arrayx = np.ravel(arrayGridx.T) 
arrayy = np.ravel(arrayGridy.T)
steerVector = np.exp(1j * (2.0*np.pi/wavelength) * (arrayx * uAngleArrival + arrayy * vAngleArrival), dtype=np.complex128)

steerVector1 = np.reshape(steerVector, (numChannels,1))

steerVector11_0 = np.reshape(steerVector, (1,numChannels))
steerVector11_1 = np.conj(steerVector11_0)   

fftSize = int(2 * (2**(np.ceil(np.log(numSamples)/np.log(2)))))

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
