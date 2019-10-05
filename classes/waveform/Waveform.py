import scipy.io
from obspy.core.utcdatetime import UTCDateTime
from ..xcorr_utils.xcorr_utils import downweight_ends
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack

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

    def __init__(self, path, filters = []):
        self._path = path
        tmp_matfile = scipy.io.loadmat(path)
        self._data =(tmp_matfile['data'][0]).flatten()
        self._delta = float(tmp_matfile['delta'][0])
        self._starttime = UTCDateTime(tmp_matfile['starttime'][0])
        self._endtime = UTCDateTime(tmp_matfile['endtime'][0])
        self._npts = tmp_matfile['npts'][0]
        self._sampling_rate = float(tmp_matfile['sampling_rate'][0])
        self._component = tmp_matfile['channel'][0][-1]
        #self._filters = filters

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

    def running_absolute_mean(self, filters, envsmooth = 1500, env_exp = 1.5, min_weight = 0.1, taper_length = 1000):
        self._data = signal.detrend(self._data, type="linear" )
        nb = np.floor(envsmooth/self._delta)
        weight = np.ones((self._data.shape[0]))
        boxc = np.ones((int(nb)))/nb
        #print boxc, boxc.shape
        nyf = (1./2)*self._sampling_rate
        print nyf
        #plt.plot(self._data)
        #plt.title("unfiltered data")
        #plt.show()
        #[b,a] = signal.butter(3,[1./100/nyf, 1./1/nyf], btype='bandpass')
        #self._data = signal.filtfilt(b,a,self._data) *  signal.tukey(self._data.shape[0],alpha = 0.05)
        for filter in filters:
            #print filter
            [b,a] = signal.butter(3,[1./filter[0]/nyf, 1./filter[1]/nyf], btype='bandpass')
            filtered_data = downweight_ends(signal.filtfilt(b,a,self._data), wlength = taper_length * self._sampling_rate) #*  signal.tukey(self._data.shape[0],alpha = 0.01)
            #plt.plot(filtered_data)
            #plt.title("filtered data")
            #plt.show()
            data_env = signal.convolve(abs(filtered_data),boxc,method="fft")
            #data_env = np.convolve(abs(filtered_data),boxc)
            data_env = data_env[boxc.shape[0]/ 2 -1: -boxc.shape[0]/ 2]
            #plt.plot(data_env)
            #plt.show()
            #print data_env.shape, self._data.shape
            data_exponent = np.power(data_env, env_exp)
            weight = weight * data_exponent / np.mean(data_exponent)
            #plt.plot(weight)
            #plt.title("weights")
            #plt.show()
            #print weight
        #weight = weight * data_exponent / mean_data_exponent
        water_level = np.mean(weight) * min_weight
        weight[weight < water_level] = water_level
        nb = 2*int(taper_length*self._sampling_rate)
        weight[:nb] = np.mean(weight)
        weight[-nb:] = np.mean(weight)
        #plt.plot(weight)
        #plt.title("final weights")
        #plt.show()
        self._data = downweight_ends((self._data / weight),wlength = taper_length * self._sampling_rate) #*  signal.tukey(self._data.shape[0],alpha = 0.1)
        #plt.plot(self._data)
        #plt.title("filtered data")
        #plt.show()
        #exit()

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
