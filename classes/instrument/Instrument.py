#!/usr/bin/python
import sys
from ..station.Station import Station
from ..waveform.Waveform import Waveform
from ..xcorr_utils import parameter_init
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
import math


class Instrument(object):
    #Station station
    #Waveform z_component
    
    def __init__(self, Station):
        self._station = Station
        self._waveforms = {}
        self._max_length = 0

    def set_filters(self, filters):
        self._filters = filters
    
    def push_waveform(self, path, component, envsmooth = 1500, env_exp = 1.5, min_weight = 0.1, 
                taper_length = 1000, plot = False, normalization = None, apply_broadband_filter = True,
                broadband_filter = [200,1], filter_order = 4):
        #print "path", path
        waveform = Waveform(path)
        self._max_length = max([waveform.get_endtime() - waveform.get_starttime(), self._max_length])
        if (len(self._waveforms) == 0):
            lat,lon,elev = waveform.get_coordinates()
            self._station.set_coordinates(lat,lon,elev)
        if component not in self._waveforms:
            self._waveforms[component] = []
        if (normalization == "RAMN"):
            waveform.running_absolute_mean(
                filters = self._filters,
                filter_order= filter_order,
                envsmooth = envsmooth,
                env_exp = env_exp,
                min_weight = min_weight, 
                taper_length = taper_length, 
                plot = False,
                apply_broadband_filter = apply_broadband_filter,
                broadband_filter = broadband_filter
            )
        elif (normalization == "BN"):
            waveform.binary_normalization()
        self._waveforms[component].append(waveform)

    def get_sampling_rate(self, component):
        return self._waveforms[component][0].get_sampling_rate()

    def get_max_length(self):
        return self._max_length

    def clear(self):
        self._waveforms = {}
    
    def get_station_coordinates(self):
        return self._station.get_coordinates()

    def get_station(self):
        return self._station

    def get_station_code(self):
        return self._station.get_station_code()

    def get_network_code(self):
        return self._station.get_network_code()

    def get_waveform(self, component, i):
        return self._waveforms[component][i]
    
    def pad_waveforms(self, component):
        for w in range(len(self._waveforms[component])):
            self._waveforms[component][w] = self.pad_waveform(self._waveforms[component][w])

    def pad_waveform(self, waveform):
        #print "max_length", self._max_length
        s = waveform.get_starttime()
        e = waveform.get_endtime()
        f = waveform.get_sampling_rate()

        diff = int((self._max_length * f)  - waveform.get_npts()) + 1
        #print "diff:",diff
        data = waveform.get_data()
        pad_end = np.zeros((diff))
        #print "pad_end.shape", pad_end.shape
        #print waveform.get_npts()
        data = np.append(data,pad_end)
        waveform.set_data(data)
        waveform.set_endtime(e + 1/f*diff)
        #print waveform.get_npts()
        waveform.recalculate_ntps()
        #print waveform.get_npts()
        return waveform
    
    def get_waveforms_mtx(self, component, l):
        l = int(l)
        num_waveforms = len(self._waveforms[component])
        cont = np.zeros((num_waveforms,l))
        if l > self._max_length * self._waveforms[component][0].get_sampling_rate():
            diff =  l - self._max_length * self._waveforms[component][0].get_sampling_rate() -1
            #print "diff:",diff
            pad_end = np.zeros((int(np.floor(diff))))
            for i in range(num_waveforms):
                cont[i,:] = np.append(self._waveforms[component][i].get_data(), pad_end)
        else:
            for i in range(num_waveforms):
                cont[i,:] = self._waveforms[component][i].get_data()[:l]
        return cont

    def get_starttime(self,component,i):
        return self._waveforms[component][i].get_starttime()

    def get_starttimes(self, component):
        num_waveforms = len(self._waveforms[component])

        cont = np.empty((1,num_waveforms),dtype = 'object')
        #print self._waveforms[component][0].get_starttime()
        for i in range(num_waveforms):
            cont[0,i] = str(self._waveforms[component][i].get_starttime())
        return cont    


    def print_waveforms(self, component, extended = False):
        if (len(self._waveforms) > 0):
            i = 0
            while i < len(self._waveforms[component]):
                self._waveforms[component][i].print_waveform(extended)
                i += 1
        else:
            print "Empty container"

