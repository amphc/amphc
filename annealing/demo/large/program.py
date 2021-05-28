import time
import numpy as np

def gen_randn(numChannels, numSamples):
    # generate noise (standard normal distribution)
    
    # use cupy to generate random numbers
    noiseReal = np.random.randn(numChannels, numSamples)
    noiseImag = np.random.randn(numChannels, numSamples)
    
    noise = (noiseReal + 1j*noiseImag) / np.sqrt(2.0)
    
    return noise

def gen_proc_datacube(weightFunction=None):
    
    # specify parameters
    
    # LFM params
    lightSpeed = 0.299792458*1e9  # speed of light
    centerFreq = 9e9              # center frequency
    sampleRate = 1000e6           # sampling rate (Hertz)
    pulseWidth = 10e-6            # pulse width (seconds)
    sweptWidth = 100e6            # swept bandwidth (Hz)
    
    # Doppler waveform params
    pri = pulseWidth / 0.1        # pri: pulse repetition interval; assume duty = 10%
    numPulses = 64                # number of pulses
    
    # target params
    targetVelocity = 30.0         # velocity (m/sec)
    targetRange0 = -250.0         # relative to range center (m)
    uAngleArrival = 0.2           # target angle of arrival
    vAngleArrival = -0.3
    targetRange = targetRange0 + pri * np.arange(numPulses) * targetVelocity
    targetT = 2.0 * targetRange / lightSpeed
    
    # array params
    wavelength = lightSpeed / centerFreq
    numPosx = 21
    numPosy = 30
    arrayPosx = np.linspace(-1.5, 1.5, num=numPosx, endpoint=True)
    arrayPosy = np.linspace(-1.05, 1.05, num=numPosy, endpoint=True)
    arrayGridx = np.tile(arrayPosx, (numPosy,1)).T
    arrayGridy = np.tile(arrayPosy, (numPosx,1))
    numChannels = numPosx * numPosy
    
    #applyWeight = False
    
    # -------------------- data cube generation begins ---------------------- #
    
    # data cube generation begins; generate Complex LFM for all the pulses
    step = 1.0/sampleRate
    timeIndices = np.arange(-pulseWidth/2.0, pulseWidth/2.0-(1.0/sampleRate)+step, step)
    numSamples = len(timeIndices)
    
    # perform pulse compression on all pulses
    # zero-padding; ensure linear convolution
    fftSize = int(2 * (2**(np.ceil(np.log(numSamples)/np.log(2)))))
    
    # compute match filter spectrum
    x0 = np.exp(1j * np.pi * sweptWidth / pulseWidth * (timeIndices**2), dtype=np.complex128)
    
    # use cupy
    #d_x0 = np.asarray(x0) # move data to current device
    d_out = np.fft.fft(x0, fftSize)
    #out = cp.asnumpy(d_out)  # move the array from device to the host
    d_matchFilter = np.conj(d_out)
    d_matchFilterShift = np.fft.fftshift(d_matchFilter)
    
    # apply weighting; set weights to 1 if weightFunction is None
    if weightFunction is None:
        d_weightFunction = np.ones(fftSize)
    else:
        d_weightFunction = np.asarray(weightFunction) # send to device
    
    d_matchFilterShift = d_matchFilterShift * d_weightFunction # element-wise multiply
    
    # apply inverse fftshift
    d_matchFilter = np.fft.fftshift(d_matchFilterShift)
    
    pulseDataNoPadding = np.zeros((numPulses, numSamples), dtype=np.complex128)
    for idx in range(numPulses):
        pulseDataNoPadding[idx,:] = np.exp(1j * np.pi * sweptWidth / pulseWidth 
                          * (timeIndices-targetT[idx])**2, dtype=np.complex128)
        
    # generate steering vector for target
    arrayx = np.ravel(arrayGridx.T) # take transpose first will produce output like order='F'
    arrayy = np.ravel(arrayGridy.T)
    #arrayx = np.reshape(arrayGridx, numChannels, order='F')
    #arrayy = np.reshape(arrayGridy, numChannels, order='F')
    #d_arrayx = np.asarray(arrayx) # send to device
    #d_arrayy = np.asarray(arrayy) # send to device
    steerVector = np.exp(1j * (2.0*np.pi/wavelength) * (arrayx * uAngleArrival 
                         + arrayy * vAngleArrival), dtype=np.complex128)
    
    # generate a received data cube: numChannels (channel) x numSamples (fast time) x numPulses (slow time)
    # potential memory allocation issue, hence the code below is slightly different
    # than the original, with dataCube allocated numChannels x numSamples only
    
    steerVector1 = np.reshape(steerVector, (numChannels,1))
    
    steerVector11 = np.reshape(steerVector, (1,numChannels))
    steerVector11 = np.conj(steerVector11)   # base on w_chan = (vec_steer)' from Matlab
    
    dataCube = np.zeros((numChannels, numSamples), dtype=np.complex128)
    beamforming = np.zeros((numPulses, numSamples), dtype=np.complex128)
    
    for idx in range(numPulses):
        pulseDataNoPadding1 = np.reshape(pulseDataNoPadding[idx,:], (1,numSamples))
        dataCube = np.multiply(steerVector1, pulseDataNoPadding1, dtype=np.complex128)
        
        # add noise to the data cube
        noise = gen_randn(numChannels, numSamples)
        dataCube = dataCube + noise
        
        # --------------------- process the data cube ----------------------- #
        
        # perform beamforming: assume weights are perfect
        beamforming[idx,:] = np.squeeze(np.matmul(steerVector11, dataCube))
        
    #del dataCube
    
    # compute spectrum (for all pulses)
    d_X = np.fft.fft(beamforming, fftSize, axis=1)
    
    # multiply in the frequency domain (prepare)
    d_matchFilter1 = np.reshape(d_matchFilter, (1,fftSize))
    d_matchFilterMultiply = np.tile(d_matchFilter1, (numPulses,1))

    # multiply in the frequency domain
    d_Y = d_X * d_matchFilterMultiply
    
    # inverse FFT
    d_y = np.fft.ifft(d_Y, axis=1)
    
    # normalize by number of time-domain samples
    d_yNorm = d_y / numSamples
    
    # apply FFT shift
    d_yNorm = np.fft.fftshift(d_yNorm, axes=1)
    
    # perform Doppler processing across all pulses
    d_ZTemp = np.fft.fft(d_yNorm, 4*numPulses, axis=0)
    d_Z = np.fft.fftshift(d_ZTemp, axes=0)
    
    return d_Z

if __name__ == '__main__':
    
    # weight can be loaded; set to None for now
    weightFunction = None

    t1 = time.time()
    Z = gen_proc_datacube(weightFunction)
    t2 = time.time()
    print('Time in second: %.6f' %(t2 - t1))
