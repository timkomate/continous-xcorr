import scipy.io
from obspy.core.utcdatetime import UTCDateTime
import numpy as np

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
        self._delta = tmp_matfile['delta'][0]
        self._starttime = UTCDateTime(tmp_matfile['starttime'][0])
        self._endtime = UTCDateTime(tmp_matfile['endtime'][0])
        self._npts = tmp_matfile['npts'][0]
        self._sampling_rate = tmp_matfile['sampling_rate'][0]
        self._component = tmp_matfile['channel'][0][-1]

    def get_data(self):
        return self._data

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
