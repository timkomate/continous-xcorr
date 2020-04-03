import glob
from ..xcorr_utils.xcorr_utils import downweight_ends, spectral_whitening
from ..instrument.Instrument import Instrument
from ..station.Station import Station
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init
from ..xcorr_utils import xcorr_utils
import numpy as np
from scipy import signal, fftpack, io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math
import json
import os


class Acorrelator(object):
    def __init__(self,component, network, station, paths, file_type = "_VEL_"):
        self._file_pattern = "{}.{}.{}{}*.mat".format(network, station, component, file_type)
        self._component = component
        self._paths = np.sort(paths)
        sta = Station(network, station, None, None, None)
        self._instrument = Instrument(sta)
        self._c = len(self._paths)
        self._offset = 0
        self._starttime_seg = np.empty((1,self._c),dtype = 'object')

    def init_matrix(self, maxlag = 600):
        self._sampling_rate = self._instrument.get_sampling_rate(
            component = self._component
        )
        shape = (self._c, int((maxlag*self._sampling_rate*2) + 1))
        self._correlations = np.zeros(shape = shape)

    def read_waveforms(self, maxlag = 600, max_waveforms = 100, filters = [], filter_order = 4,
                    envsmooth = 1500, env_exp = 1.5, min_weight = 0.1, taper_length = 1000, 
                    plot = False, apply_broadband_filter_tdn = False, 
                    broadband_filter_tdn = [200,1]):
        if (parameter_init.binary_normalization):
            self._normalization_method = "BN"
        elif (parameter_init.running_absolute_mean_normalization):
            self._normalization_method = "RAMN"
        else:
            self._normalization_method = "WN"
        if (parameter_init.apply_spectral_whitening):
            self._whitening = "W"
        else:
            self._whitening = ""
        self._instrument.clear()
        self._instrument.set_filters(filters)
        if (max_waveforms > 0):
            self._max_waveforms = max_waveforms
        else:
            self._max_waveforms = self._c
        i = 0
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            file = glob.glob("{}/{}".format(self._paths[i + self._offset], self._file_pattern))[0]
            self._instrument.push_waveform(
                path = file,
                component = self._component,
                envsmooth = envsmooth,
                env_exp = env_exp,
                min_weight = min_weight,
                taper_length = taper_length,
                plot = plot,
                normalization = self._normalization_method,
                apply_broadband_filter =apply_broadband_filter_tdn,
                broadband_filter = broadband_filter_tdn,
                filter_order= filter_order
            )
            i += 1
        if (self._offset == 0):
            self.init_matrix(maxlag)

    def acorr(self, maxlag = 600, spectrumexp = 0.7, 
            espwhitening = 0.05, taper_length_whitening = 100, 
            verbose = False, apply_broadband_filter = False,
            broadband_filter = [200,1], filter_order = 4):
        i = 0
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            j = i + self._offset
            #print i, j
            a = self._instrument.get_waveform(component = self._component, 
                i = i
            ).get_data()

            starttime = str(self._instrument.get_starttime(
                component = self._component,
                i = i
            ))
            self._starttime_seg[0,j] = starttime

            acf = signal.correlate(a,a, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*self._sampling_rate)[0]
            self._lagtime = tcorr[dN] * (1. / self._sampling_rate)
            acf = acf[dN]

            self._correlations[j,:] = acf
            i += 1
        self._stacked_acf = np.sum(self._correlations, axis=0)
        self._offset += self._max_waveforms

    def acorr_pcc(self, maxlag = 600, spectrumexp = 0.7, 
            espwhitening = 0.05, taper_length_whitening = 100, 
            verbose = False, apply_broadband_filter = False,
            broadband_filter = [200,1], filter_order = 4):
        i = 0
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            j = i + self._offset
            #print i, j
            a = self._instrument.get_waveform(component = self._component, 
                i = i
            ).get_data()

            starttime = str(self._instrument.get_starttime(
                component = self._component,
                i = i
            ))
            self._starttime_seg[0,j] = starttime

            self._lagtime, acf = xcorr_utils.apcc2(a,1/self._sampling_rate,-maxlag,maxlag)
            #print self._lagtime
            self._correlations[j,:] = acf
            i += 1
        #self._stacked_acf = np.sum(self._correlations, axis=0)
        self._offset += self._max_waveforms
    
    def calculate_linear_stack(self):
        self._stacked_acf = np.sum(self._correlations, axis=0) / self._c
        #self._simmetric_part, self._simmetric_lagtime = self.calculate_simmetric_part()

    def save_acf(self, path, tested_parameter = "", extended_save = True):
        compflag = self._component
        corrflag = "ACF"
        nstack = self._c
        network = self._instrument.get_network_code()
        station = self._instrument.get_station_code()
        save_path = "{}/{}_{}_{}_{}_{}_{}{}{}".format(
            path, corrflag, network, station,
            compflag, nstack, self._normalization_method, 
            self._whitening, tested_parameter
        )
        if not os.path.exists(path):
            os.makedirs(path)
        if (extended_save):
            matfile = {
                "compflag" : compflag,
                "corrflag" : corrflag,
                "cross12" : self._stacked_acf,
                "cutvec" : self._correlations,
                "dtnew" : 1./self._sampling_rate,
                "lagsx1x2" : self._lagtime,
                "nstack" : nstack,
                "Station1" :station,
                "starttime_seg" : self._starttime_seg
            }
        else:
            matfile = {
                "compflag" : compflag,
                "corrflag" : corrflag,
                "cross12" : self._stacked_acf,
                "dtnew" : 1./self._sampling_rate,
                "lagsx1x2" : self._lagtime,
                "nstack" : nstack,
                "Station1" :station,
                "starttime_seg" : self._starttime_seg
            }
        io.savemat(save_path, matfile)
        return save_path
    
    def get_nstack(self):
        return self._c
