import scipy.io
from obspy.core.utcdatetime import UTCDateTime
import numpy as np
import matplotlib.pyplot as plt

class Waveform(object):
    #private:
        #string path
        #double[] data;
        #double delta;
        #datetime starttime;
        #datetime endtime;
        #int npts;
        #double sampling_rate;
        #string component;

    def __init__(self, path):
        self._path = path
        tmp_matfile = scipy.io.loadmat(path)
        self._data =(tmp_matfile['data'][0]).flatten()
        self._delta = float(tmp_matfile['delta'][0])
        self._starttime = UTCDateTime(tmp_matfile['starttime'][0])
        self._endtime = UTCDateTime(tmp_matfile['endtime'][0])
        self._npts = tmp_matfile['npts'][0]
        self._sampling_rate = float(tmp_matfile['sampling_rate'][0])
        self._component = tmp_matfile['channel'][0][-1]

    def get_data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def get_starttime(self):
        return self._starttime

    def set_starttime(self, starttime):
        self._starttime = starttime

    def get_endtime(self):
        return self._endtime

    def set_endtime(self, endtime):
        self._endtime = endtime
    
    def get_npts(self):
        #return self._data.shape[0]
        return self._npts

    def set_npts(self, npts):
        self._npts = npts
    
    def recalculate_ntps(self):
        self._npts = self._data.shape[0]

    def get_sampling_rate(self):
        return self._sampling_rate   

    def get_dt(self):
        return self._delta     

    def plot(self):
        plt.plot(self._data)
        plt.show()

    def binary_normalization(self):
        A = self._data > 0
        self._data[A] = 1
        self._data[np.invert(A)] = -1

    def print_waveform(self,extended = False):
        print "Path:", self._path
        if extended:
            print "Data:", self._data
        print "Delta t:", self._delta
        print "Start time:", self._starttime
        print "End time:", self._endtime
        print "Number of samples:", self._npts
        print "Sampling frequency:", self._sampling_rate
        print "Component:", self._component
