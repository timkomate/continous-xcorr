import scipy.io
from obspy.core.utcdatetime import UTCDateTime
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

    def running_absolute_mean(self, filters, envsmooth = 1500, env_exp = 1.5, min_weight = 0.1):
        nb = np.floor(envsmooth/self._delta)
        weight = np.ones((self._data.shape[0]))
        boxc = np.ones((int(nb)))/nb
        #boxc =  np.pad(boxc, (0,np.abs(self._data.shape[0] - boxc.shape[0])),mode="constant", constant_values=(0))
        print boxc, boxc.shape
        nyf = 1/(2*(1./self._sampling_rate))
        plt.plot(self._data)
        plt.title("unfiltered data")
        plt.show()
        for filter in filters:
            print filter
            [b,a] = signal.butter(3,[1./filter[0]/nyf, 1./filter[1]/nyf], btype='bandpass')
            filtered_data = signal.filtfilt(b,a,self._data) #*  signal.tukey(self._data.shape[0],alpha = 0.75)
            plt.plot(filtered_data)
            plt.title("filtered data")
            plt.show()
            data_env = signal.convolve(abs(filtered_data),boxc,method="fft")
            #data_env = np.convolve(abs(filtered_data),boxc)
            data_env = data_env[boxc.shape[0]/ 2 -1: -boxc.shape[0]/ 2]
            plt.plot(data_env)
            plt.show()
            print data_env.shape, self._data.shape
            data_exponent = np.power(data_env, env_exp)
            mean_data_exponent = np.mean(data_exponent)
            weight = weight * data_exponent / mean_data_exponent
            plt.plot(weight)
            plt.title("weights")
            plt.show()
            print weight
        weight[weight < min_weight * mean_data_exponent] = min_weight * mean_data_exponent
        self._data = (self._data / weight) *  signal.tukey(self._data.shape[0],alpha = 0.1)
        plt.plot(self._data)
        plt.title("filtered data")
        plt.show()
        #whitened = whitened * signal.tukey(len(whitened))
        exit()

    def running_absolute_mean2(self, envsmooth = 1500, env_exp = 1, min_weight = 0.1):
        nb = np.floor(envsmooth/self._delta)
        weight = np.ones((self._data.shape[0]))
        boxc = np.ones((nb))/nb
        M = boxc.shape[0] + self._data.shape[0] - 1
        #boxc =  np.pad(boxc, (0,np.abs(self._data.shape[0] - boxc.shape[0])),mode="constant", constant_values=(0))
        #print boxc, boxc.shape
        #data_env = signal.convolve(abs(self._data),boxc)
        print np.fft.fft(boxc, M)
        data_env = np.fft.ifft(np.fft.fft(self._data, M)*np.fft.fft(boxc, M))
        #data_env = fftpack.ifft(fftpack.fft(self._data, M)*fftpack.fft(boxc, M))
        print data_env
        data_env = data_env[boxc.shape[0]/ 2 -1 : -boxc.shape[0]/ 2]
        #print data_env.shape, self._data.shape
        data_exponent = np.power(data_env, env_exp)
        mean_data_exponent = np.mean(data_exponent)
        weight = weight * data_exponent / mean_data_exponent
        weight[weight < min_weight * mean_data_exponent] = min_weight * mean_data_exponent
        self._data = self._data / weight

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
