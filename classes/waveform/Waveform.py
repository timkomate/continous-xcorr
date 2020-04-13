import scipy.io
from obspy.core.utcdatetime import UTCDateTime
from ..xcorr_utils.xcorr_utils import downweight_ends
#from ..xcorr_utils import parameter_init
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

    def __init__(self, path):

        self._path = path
        tmp_matfile = scipy.io.loadmat(path)
        self._data =(tmp_matfile['data'][0]).flatten()
        self._lat = float(tmp_matfile['lat'][0])
        self._lon = float(tmp_matfile['lon'][0])
        self._elev = float(tmp_matfile['elevation'][0])
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

    def get_coordinates(self):
        return [self._lat, self._lon, self._elev]

    def get_endtime(self):
        return self._endtime

    def set_endtime(self, endtime):
        self._endtime = endtime
    
    def get_npts(self):
        return self._data.shape[0]
        #return self._npts

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
        #self._data[A] = 1
        #self._data[np.invert(A)] = -1
        self.data = A.astype(int).flatten()

    def running_absolute_mean(self, filters, filter_order = 4, envsmooth = 1500, 
                    env_exp = 1.5, min_weight = 0.1, taper_length = 1000, plot = False,
                    apply_broadband_filter = True, broadband_filter = [200,1]):
    
        data = (signal.detrend(self._data, type="linear" )) #/ np.power(10,9)
        nb = np.floor(envsmooth/self._delta)
        weight = np.ones((data.shape[0]))
        boxc = np.ones((int(nb)))/nb
        nyf = (1./2)*self._sampling_rate
        if (plot):
            plt.plot(self._data)
            plt.title("unfiltered data")
            plt.show()
        if (apply_broadband_filter):
            [b,a] = signal.butter(
                N = filter_order,
                Wn = [1./broadband_filter[0]/nyf, 1./broadband_filter[1]/nyf], 
                btype='bandpass'
            )
            data = signal.filtfilt(
                b = b,
                a = a,
                x = self._data
            )
        for filter in filters:
            [b,a] = signal.butter(
                N = filter_order,
                Wn = [1./filter[0]/nyf, 1./filter[1]/nyf], 
                btype='bandpass'
            )
            filtered_data = signal.filtfilt(
                b = b,
                a = a,
                x = self._data
            )
            filtered_data = downweight_ends(
                data = filtered_data, 
                wlength = taper_length * self._sampling_rate
            )
            if (plot):
                plt.plot(filtered_data)
                plt.title("filtered data")
                plt.show()

            data_env = signal.convolve(
                in1 = abs(filtered_data),
                in2 = boxc,
                method="fft"
            )
            data_env = data_env[boxc.shape[0]/ 2 -1: -boxc.shape[0]/ 2]
            if (plot):
                plt.plot(data_env)
                plt.title("envelope")
                plt.show()
            data_exponent = np.power(data_env, env_exp)
            weight = weight * data_exponent / np.mean(data_exponent)
            if (plot):
                plt.plot(weight)
                plt.title("weights")
                plt.show()
        water_level = np.mean(weight) * min_weight
        weight[weight < water_level] = water_level
        nb = 2*int(taper_length*self._sampling_rate)
        weight[:nb] = np.mean(weight)
        weight[-nb:] = np.mean(weight)
        if (plot):
            plt.plot(weight)
            plt.title("final weights")
            plt.show()
        if (plot):
            plt.plot(data)
            plt.title("filtered data")
            plt.show()
        self._data = downweight_ends(
            data = (self._data / weight),
            wlength = taper_length *self._sampling_rate
        )

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
