import glob
from ..xcorr_utils.xcorr_utils import downweight_ends
from ..instrument.Instrument import Instrument
from ..station.Station import Station
from ..xcorr_utils.setup_logger import logger
import numpy as np
from scipy import signal, fftpack, io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math
import json


class Xcorrelator(object):
    def __init__(self,component1, network1, station1, component2, network2, station2, paths, json_path):
        self._inst1 = "%s.%s.%s_*.mat" % (network1, station1, component1)
        self._inst2 = "%s.%s.%s_*.mat" % (network2, station2, component2)
        self._paths = np.sort(paths)
        station_data_dict = Xcorrelator.load_station_infos(json_path)
        [lat1, lon1, elev1] = Xcorrelator.find_station_coordinates(station_data_dict, network1, station1)
        [lat2, lon2, elev2] = Xcorrelator.find_station_coordinates(station_data_dict, network2, station2)
        sta1 = Station(network1, station1, lat1, lon1, elev1)
        sta2 = Station(network2, station2, lat2, lon2, elev2)
        self._distance = Xcorrelator.calc_distance_km(sta1.get_coordinates(), sta2.get_coordinates())
        self._instrument1 = Instrument(sta1)
        self._instrument2 = Instrument(sta2)
        self._c = 0

    def read_waveforms(self, filters = [], envsmooth = 1500, env_exp = 1.5, 
                min_weight = 0.1, taper_length = 1000, plot = False):
        #print "Reading dataset..."
        self._normalization_method = "RAMN" if len(filters) > 0 else "BN"
        self._instrument1.clear()
        self._instrument2.clear()
        #start = timer()
        self._c = len(self._paths)
        self._instrument1.set_filters(filters)
        self._instrument2.set_filters(filters)
        for path in self._paths:
            file1 = glob.glob(path + "/" + self._inst1)[0]
            file2 = glob.glob(path + "/" + self._inst2)[0]
            self._instrument1.push_waveform(
                path = file1,
                envsmooth = envsmooth,
                env_exp = env_exp,
                min_weight = min_weight,
                taper_length = taper_length,
                plot = plot
            )
            self._instrument2.push_waveform(
                path = file2,
                envsmooth = envsmooth,
                env_exp = env_exp,
                min_weight = min_weight,
                taper_length = taper_length,
                plot = plot
            )
        #end = timer()
        #print "Reading dataset and time-domain normalization:", end - start, "seconds\n"

    def xcorr(self, maxlag = 600, spectrumexp = 0.7, 
            espwhitening = 0.05, taper_length = 100, verbose = False):
        #print "Cross-correlation..."
        i = 0
        #start = timer()
        self._sampling_rate = self._instrument1.get_sampling_rate()
        shape = (self._c, int((maxlag*self._sampling_rate*2) + 1))
        self._xcorrelations = np.zeros(shape = shape)
        while i < self._c:
            a = self._instrument1.get_waveform(i).get_data()
            b = self._instrument2.get_waveform(i).get_data()
            ccf = signal.correlate(a,b, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*self._sampling_rate)[0]
            self._lagtime = tcorr[dN] * (1. / self._sampling_rate)
            ccf = ccf[dN]
            ccf = self.spectral_whitening(
                ccf = ccf, 
                spectrumexp = spectrumexp, 
                espwhitening = 0.05,
                taper_length = 100,
                plot = verbose,
            )
            self._xcorrelations[i,:] = ccf
            i += 1
        #end = timer()
        #print "Cross-correlation:", end - start, "seconds\n"
        self._stacked_ccf = np.sum(self._xcorrelations, axis=0)
        self._simmetric_part, self._simmetric_lagtime = self.calculate_simmetric_part()
    
    def calculate_simmetric_part(self):
        size = self._stacked_ccf.size
        return self._stacked_ccf[size/2 + 1:] + np.flipud(self._stacked_ccf[0:size/2]), self._lagtime[size/2 + 1:]

    def save_figures(self,path):
        plt.imshow(self._xcorrelations / self._xcorrelations.max(axis = 1)[:,np.newaxis], aspect = "auto",  cmap = "bone")
        plt.savefig("%s/daily_ccfs.png" % path)
        
        plt.plot(self._stacked_ccf)
        plt.savefig("%s/stacked_ccf.png" % path)

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

    def spectral_whitening(self, ccf, spectrumexp = 0.7, espwhitening = 0.05, taper_length = 100, plot = False):
        '''
        apply spectral whitening to np.array ccf, divide spectrum of data1 by its smoothed version
    
        data1: np.array, time series vector
        wlen: int or None (default), length of boxcar for smoothing of spectrum, number of (spectral) samples
    
        return:
            np.array, spectrally whitened time series vector
        '''
        if (plot):
            plt.plot(ccf)
            plt.title("original dataset")
            plt.show()
        spectrum =(np.fft.rfft(signal.detrend(ccf,type="linear")))
        spectrum_abs = np.abs(spectrum)
        if (plot):
            f = np.fft.rfftfreq(len(ccf), d=1./self._sampling_rate)
            plt.plot(f,spectrum)
            plt.plot(f,spectrum_abs)
            plt.title("specrtum and ampl. spectrum")
            plt.show()
        water_level = np.mean(spectrum_abs) * espwhitening
        spectrum_abs[(spectrum_abs < water_level)] = water_level
        
        if (plot):
            plt.plot(f,spectrum_abs)
            plt.title("spectrum after water level")
            plt.show()
            fig, axs = plt.subplots(3)
            fig.suptitle('Vertically stacked subplots')
            axs[0].plot(spectrum)
            axs[1].plot(np.power(spectrum_abs,spectrumexp))
            axs[2].plot(spectrum_abs)
            plt.show()
        
        #whitening
        spectrum = np.divide(spectrum, (np.power(spectrum_abs,spectrumexp)))
        #spectrum = downweight_ends(spectrum, wlength = (taper_length * self._sampling_rate))
        spectrum[0] = 0

        if (plot):
            plt.plot(f,np.abs(spectrum))
            plt.title("spectrum after whitening")
            plt.show()

        whitened = np.fft.irfft((spectrum))
        whitened = signal.detrend(whitened,type="linear")
        whitened =  downweight_ends(whitened,wlength= (taper_length * self._sampling_rate))

        nyf = (1./2)*self._sampling_rate
        [b,a] = signal.butter(3,[(1./100)/nyf,(1./1)/nyf], btype='bandpass')
        whitened = signal.filtfilt(b,a,whitened)
        if (plot):
            plt.plot(whitened)
            plt.title("whitened signal after filtering")
            plt.show()
        #remove mean
        whitened = whitened * np.mean(np.abs(whitened))
        whitened = np.append(whitened, 0)
        return whitened

    def save_ccf(self, path, tested_parameter = "", extended_save = True):
        compflag = "ZZ"
        corrflag = "CCF"
        nstack = self._c
        station1 = self._instrument1.get_station_code()
        station2 = self._instrument2.get_station_code()
        save_path = "%s/%s_%s_%s_%s_%s_%s%s" % (path,corrflag,station1,station2,compflag,nstack, self._normalization_method, tested_parameter)
        if (extended_save):
            matfile = {
                "compflag" : compflag,
                "corrflag" : corrflag,
                "cross12" : self._stacked_ccf,
                "cutvec" : self._xcorrelations,
                "Dist" : self._distance * 1000,
                "dtnew" : 1./self._sampling_rate,
                "lagsx1x2" : self._lagtime,
                "nstack" : nstack,
                "Station1" :station1,
                "Station2" : station2
            }
        else:
            matfile = {
                "compflag" : compflag,
                "corrflag" : corrflag,
                "cross12" : self._stacked_ccf,
                "Dist" : self._distance * 1000,
                "dtnew" : 1./self._sampling_rate,
                "lagsx1x2" : self._lagtime,
                "nstack" : nstack,
                "Station1" :station1,
                "Station2" : station2
            }
        io.savemat(save_path, matfile)
        #print "File has been saved as %s" % (save_path)
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
