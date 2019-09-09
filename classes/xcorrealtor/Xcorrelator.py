import glob
from ..instrument.Instrument import Instrument
from ..station.Station import Station
import numpy as np
from scipy import signal, fftpack
import matplotlib.pyplot as plt
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
        while i < self._c:
            a = self._instrument1.get_waveform(i).get_data()
            b = self._instrument2.get_waveform(i).get_data()
            print "a size:",np.size(a)
            print "b size:",np.size(b)
            c = signal.correlate(a,b, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*5)[0]
            print tcorr, tcorr.shape
            print dN, dN.shape
            c = c[dN]
            plt.plot(c)
            plt.show()
            print "c size:", np.size(c)
            
            
            i += 1

    def correct_waveform_lengths(self):
        i = 0
        while i < self._c:
            a = self._instrument1.get_waveform(i)
            b = self._instrument2.get_waveform(i)
            dt = a.get_dt()
            s1 = a.get_starttime()
            s2 = b.get_starttime()
            e1 = a.get_endtime()
            e2 = b.get_endtime()
            diff_start = math.floor(abs(s1-s2) / dt)
            diff_end = math.ceil(abs(e1-e2) / dt)
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
            else:
                b.set_endtime(e2 - (diff_end*dt))
                data = b.get_data()
                data = data[:-diff_end]
                b.set_data(data)
            a.recalculate_ntps()
            b.recalculate_ntps()
            #a.print_waveform()
            #b.print_waveform()
            i += 1

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