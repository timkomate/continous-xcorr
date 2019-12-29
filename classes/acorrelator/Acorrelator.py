import glob
from ..xcorr_utils.xcorr_utils import downweight_ends, spectral_whitening
from ..instrument.Instrument import Instrument
from ..station.Station import Station
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init
import numpy as np
from scipy import signal, fftpack, io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math
import json


class Acorrelator(object):
    def __init__(self,component, network, station, paths):
        self._file_pattern = "%s.%s.%s*.mat" % (network, station, component)
        self._component = component
        self._paths = np.sort(paths)
        sta = Station(network, station, None, None, None)
        self._instrument = Instrument(sta)
        self._c = len(self._paths)
        self._offset = 0

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
        #start = timer()
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            j = i + self._offset
            print i, j
            a = self._instrument.get_waveform(component = self._component, 
                i = i
            ).get_data()
            acf = signal.correlate(a,a, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*self._sampling_rate)[0]
            self._lagtime = tcorr[dN] * (1. / self._sampling_rate)
            acf = acf[dN]
            acf = spectral_whitening(
                data = acf,
                sampling_rate = self._sampling_rate,
                spectrumexp = spectrumexp,
                espwhitening = espwhitening,
                taper_length = taper_length_whitening,
                apply_broadband_filter = apply_broadband_filter,
                broadband_filter = broadband_filter,
                filter_order = filter_order,
                plot = verbose,
            )
            self._correlations[i,:] = acf
            i += 1
        self._stacked_acf = np.sum(self._correlations, axis=0)
        self._offset += self._max_waveforms

    def save_figures(self,path):
        plt.imshow(self._xcorrelations / self._xcorrelations.max(axis = 1)[:,np.newaxis], aspect = "auto",  cmap = "bone")
        plt.savefig("%s/daily_ccfs.png" % path)
        
        plt.plot(self._stacked_ccf)
        plt.savefig("%s/stacked_ccf.png" % path)

    def save_acf(self, path, tested_parameter = "", extended_save = True):
        compflag = self._component
        corrflag = "ACF"
        nstack = self._c
        station = self._instrument.get_station_code()
        save_path = "%s/%s_%s_%s_%s_%s%s" % (path,corrflag,station,compflag,nstack, self._normalization_method, tested_parameter)
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
            }
        io.savemat(save_path, matfile)
        return save_path
        

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
            fig, axs = plt.subplots(3)
            axs[0].plot(fft1)
            axs[1].plot(fft2)
            axs[2].plot(fft1-fft2)
            plt.show()
            i += 1

    @staticmethod
    def calc_distance_deg(s_coordinates, e_coordinates):
        slat = s_coordinates[0]
        slon = s_coordinates[1]
        elat = e_coordinates[0]
        elon = e_coordinates[1]
        FLATTENING = 0.00335281066474
        f = (1. - FLATTENING) * (1. - FLATTENING)
        geoc_elat = math.atan(f * math.tan((math.pi/180) * elat))
        celat = math.cos(geoc_elat)
        selat = math.sin(geoc_elat)
        geoc_slat = math.atan(f * math.tan((math.pi/180) * slat))
        rdlon = (math.pi/180) * (elon - slon)
        cslat = math.cos(geoc_slat)
        sslat = math.sin(geoc_slat)
        cdlon = math.cos(rdlon)
        sdlon = math.sin(rdlon)
        cdel = sslat * selat + cslat * celat * cdlon
        cdel if cdel<1 else 1
        cdel if cdel>-1 else -1
        delta = (180/math.pi) * math.acos(cdel)
        return delta
    
    @staticmethod
    def calc_distance_km(s_coordinates, e_coordinates):
        slat = math.radians(s_coordinates[0])
        slon = math.radians(s_coordinates[1])
        elat = math.radians(e_coordinates[0])
        elon = math.radians(e_coordinates[1])
        R = 6371.0
        dlon = slon - elon
        dlat = slat - elat
        a = math.sin(dlat / 2)**2 + math.cos(elat) * math.cos(slat) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
    
    @staticmethod
    def load_station_infos(json_path):
        with open(json_path, 'r') as fp:
            data = json.load(fp)
        return data
    
    @staticmethod
    def find_station_coordinates(data, network, station):
        elev = data[network][station]["elevation"]
        lat = data[network][station]["latitude"]
        lon = data[network][station]["longitude"]
        return [lat, lon, elev]


#TODO
#The stripped waveforms sometimes not equally long. Needs to check
