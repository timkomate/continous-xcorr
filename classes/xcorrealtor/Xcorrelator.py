import glob
from ..instrument.Instrument import Instrument
from ..station.Station import Station
import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
from obspy.signal.util import next_pow_2
from timeit import default_timer as timer
import math

class Xcorrelator(object):
    def __init__(self,component1, network1, station1, component2, network2, station2, paths):
        self._inst1 = "%s.%s.%s_*.mat" % (network1, station1, component1)
        self._inst2 = "%s.%s.%s_*.mat" % (network2, station2, component2)
        self._paths = paths
        sta = Station("HU", "BUD", 47, 19, 165)
        self._instrument1 = Instrument(sta)
        self._instrument2 = Instrument(sta)
        self._c = 0

    def read_waveforms(self):
        for path in self._paths:
            self._c += 1
            file1 = glob.glob(path + "/" + self._inst1)[0]
            file2 = glob.glob(path + "/" + self._inst2)[0]
            print file1, file2
            self._instrument1.push_waveform(file1)
            self._instrument2.push_waveform(file2)

    def xcorr(self,maxlag):
        i = 0
        start = timer()
        while i < self._c:
            print i
            a = self._instrument1.get_waveform(i).get_data()
            b = self._instrument2.get_waveform(i).get_data()
            print self._instrument1.get_waveform(i).print_waveform()
            print self._instrument2.get_waveform(i).print_waveform()
            print "a size:",np.size(a)
            print "b size:",np.size(b)
            #plt.plot(a)
            #plt.show()
            c = signal.correlate(a,b, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*5)[0]
            #print tcorr, tcorr.shape
            #print dN, dN.shape
            c = c[dN]
            #plt.plot(c)
            #plt.show()
            print "c size:", np.size(c)
            self.spectral_whitening(c)
            
            i += 1
        end = timer()
        print(end - start)

    def correct_waveform_lengths(self):
        i = 0
        while i < self._c:
            a = self._instrument1.get_waveform(i)
            b = self._instrument2.get_waveform(i)
            a.print_waveform()
            b.print_waveform()
            dt = a.get_dt()
            s1 = a.get_starttime()
            s2 = b.get_starttime()
            e1 = a.get_endtime()
            e2 = b.get_endtime()
            diff_start = int(math.floor(abs(s1-s2) / dt))
            diff_end = int(math.ceil(abs(e1-e2) / dt))
            #s1 started earlier
            if s1 < s2: 
                a.set_starttime(s1 + (diff_start*dt))
                data = a.get_data()
                data = data[diff_start:]
                a.set_data(data)
            else:
                b.set_starttime(s2 + (diff_start*dt))
                data = b.get_data()
                data = data[diff_start:]
                b.set_data(data)
            if e1 > e2: 
                a.set_endtime(e1 - (diff_end*dt))
                data = a.get_data()
                data = data[:-diff_end]
                a.set_data(data)
            elif (diff_end):
                b.set_endtime(e2 - (diff_end*dt))
                data = b.get_data()
                data = data[:-diff_end]
                b.set_data(data)
            a.recalculate_ntps()
            b.recalculate_ntps()
            i += 1

    def spectral_whitening(self,data1, wlen=None, spectrumexp = 1):
        '''
        apply spectral whitening to np.array data1, divide spectrum of data1 by its smoothed version
    
        data1: np.array, time series vector
        wlen: int or None (default), length of boxcar for smoothing of spectrum, number of (spectral) samples
            if None, 1% of nfft will be used
    
        return:
            np.array, spectrally whitened time series vector
        '''
        ndat = len(data1)
        #nfft = next_pow_2(2*ndat)
        plt.plot(data1)
        plt.show()
        s1 =(fftpack.rfft(data1)) # real part, length nfft/2+1
        f = fftpack.rfftfreq(ndat, d=0.2)
        print s1, type(s1), s1.shape
        print f, type(f), f.shape
        plt.plot(f,s1)
        plt.show()
        dada = fftpack.irfft(s1)
        plt.plot(dada)
        plt.show()

        plt.plot(data1-dada)
        plt.show()

        #s2 = fftpack.fft(data1)

        #print s2, type(s2), s2.shape
        #plt.plot(s2)
        #plt.show()

        #plt.plot(s1-s2)
        #plt.show()
        # winlen is no of samples of smoothing boxcar
        # ... winlen should be max nfft/10
        #winlen = int(nfft/100)
        #if wlen is not None:
        #    winlen = min(wlen, winlen)
    
        #s1s = np.convolve(abs(s1), np.ones(winlen)/winlen, 'same') # smoothed spectrum
        #s1s = fftconvolve(abs(s1), np.ones(winlen)/winlen, 'same') # smoothed spectrum
        ## fftconv not faster than np.convolve here
        #s1s = fftconv(abs(s1), np.ones(winlen)/winlen, nfft) # smoothed spectrum
        
        # waterlevel smoothed spectrum
        #s1s[(s1s < 1E-10)] = 1E-10
        
        #s1w = s1 / np.power(s1s, spectrumexp) # whitened spectrum
        #x1w = np.fft.irfft(s1w, nfft)[:ndat] # IFFT -> data after spectral whitening
        #return x1w

    def fft(self):
        i = 0
        while i < self._c:
            a = self._instrument1.get_waveform(i).get_data()
            print type(a)
            start = timer()
            fft1 = fftpack.fft(a)
            fft2 = np.fft.fft(a)
            end = timer()
            print (end - start)
            #plt.plot(fft)
            fig, axs = plt.subplots(3)
            axs[0].plot(fft1)
            axs[1].plot(fft2)
            axs[2].plot(fft1-fft2)
            plt.show()
            i += 1
