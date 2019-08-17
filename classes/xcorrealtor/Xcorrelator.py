import glob
from ..instrument.Instrument import Instrument
from ..station.Station import Station
import numpy as np
from scipy import signal

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

    def xcorr(self):
        i = 0
        while i < self._c:
            a = self._instrument1.get_waveform(i)
            b = self._instrument2.get_waveform(i)
            print a
            print "B"
            print b
            a = a.get_data()
            b = b.get_data()
            #corr = signal.correlate(a,b, mode='full', method='fft')
            c = signal.correlate(a,b, mode = 'full', method="fft")
            i += 1