import glob
from ..xcorr_utils.xcorr_utils import downweight_ends, spectral_whitening
from ..instrument.Instrument import Instrument
from ..station.Station import Station
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init
import numpy as np
from scipy import signal, fftpack, io
import os
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math
import json


class Xcorrelator(object):
    def __init__(self,component1, network1, station1, component2, network2, station2, paths, file_type = "_VEL_"):
        self._inst1 = "{}.{}.{}{}*.mat".format(network1, station1, component1, file_type)
        self._inst2 = "{}.{}.{}{}*.mat".format(network2, station2, component2, file_type)
        self._paths = np.sort(paths)
        self._component1 = component1
        self._component2 = component2
        sta1 = Station(network1, station1, None, None, None)
        sta2 = Station(network2, station2, None, None, None)
        #self._distance = Xcorrelator.calc_distance_km(sta1.get_coordinates(), sta2.get_coordinates())
        self._instrument1 = Instrument(sta1)
        self._instrument2 = Instrument(sta2)
        self._c = len(self._paths)
        self._offset = 0
        self._starttime_seg = np.empty((1,self._c),dtype = 'object')
    
    def init_matrix(self, maxlag = 600):
        self._sampling_rate = self._instrument1.get_sampling_rate(
            component = self._component1
        )
        shape = (self._c, int((maxlag*self._sampling_rate*2) + 1))
        self._xcorrelations = np.zeros(shape = shape)
        self._distance = Xcorrelator.calc_distance_km(
            s_coordinates = self._instrument1.get_station_coordinates(), 
            e_coordinates = self._instrument2.get_station_coordinates()
        )

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
        self._instrument1.clear()
        self._instrument2.clear()
        self._instrument1.set_filters(filters)
        self._instrument2.set_filters(filters)
        if (max_waveforms > 0):
            self._max_waveforms = max_waveforms
        else:
            self._max_waveforms = self._c
        i = 0
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            file1 = glob.glob("{}/{}".format(self._paths[i + self._offset], self._inst1))[0]
            file2 = glob.glob("{}/{}".format(self._paths[i + self._offset], self._inst2))[0]
            #print file1, file2
            self._instrument1.push_waveform(
                path = file1,
                component = self._component1,
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
            self._instrument2.push_waveform(
                path = file2,
                component = self._component2,
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
        #raw_input()

    def xcorr(self, maxlag = 600, spectrumexp = 0.7, 
            espwhitening = 0.05, taper_length_whitening = 100, 
            verbose = False, apply_broadband_filter = False,
            broadband_filter = [200,1], filter_order = 4):
        i = 0
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            j = i + self._offset
            #print i, j
            a = self._instrument1.get_waveform(
                component = self._component1,
                i = i
            ).get_data()
            b = self._instrument2.get_waveform(
                component = self._component2,
                i = i
            ).get_data()
            
            starttime = str(self._instrument1.get_starttime(
                component = self._component1,
                i = i
            ))
            #print starttime, rem_waveform

            self._starttime_seg[0,j] = starttime
            #print self._starttime_seg
            ccf = signal.correlate(a,b, mode = "full", method="fft")
            tcorr = np.arange(-a.shape[0] + 1, a.shape[0])
            dN = np.where(np.abs(tcorr) <= maxlag*self._sampling_rate)[0]
            self._lagtime = tcorr[dN] * (1. / self._sampling_rate)
            ccf = ccf[dN]
            ccf = spectral_whitening(
                data = ccf,
                sampling_rate = self._sampling_rate,
                spectrumexp = spectrumexp,
                espwhitening = espwhitening,
                taper_length = taper_length_whitening,
                apply_broadband_filter = apply_broadband_filter,
                broadband_filter = broadband_filter,
                filter_order = filter_order,
                plot = verbose,
            )
            self._xcorrelations[j,:] = ccf
            i += 1
        self._stacked_ccf = np.sum(self._xcorrelations, axis=0) #/ self._c
        self._simmetric_part, self._simmetric_lagtime = self.calculate_simmetric_part()
        self._offset += self._max_waveforms
    
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
        #start = timer()
        rem_waveform = self._max_waveforms if self._c - self._offset > self._max_waveforms else self._c - self._offset
        while i < rem_waveform:
            #print i
            a = self._instrument1.get_waveform(self._component1, i)
            b = self._instrument2.get_waveform(self._component2, i)
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

    def save_ccf(self, path, tested_parameter = "", extended_save = True):
        compflag = "ZZ"
        corrflag = "CCF"
        nstack = self._c
        station1 = self._instrument1.get_station_code()
        station2 = self._instrument2.get_station_code()
        save_path = "%s/%s_%s_%s_%s_%s_%s%s" % (path,corrflag,station1,station2,compflag,nstack, self._normalization_method, tested_parameter)
        if not os.path.exists(path):
            os.makedirs(path)
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
                "Station2" : station2,
                "starttime_seg" : self._starttime_seg
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
                "Station2" : station2,
                "starttime_seg" : self._starttime_seg
            }
        io.savemat(save_path, matfile)
        print "File has been saved as {}".format(save_path)
        return save_path
        
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
#Check if the sampling rate is the same for both waveforms