#!/usr/bin/python
import sys
from ..station.Station import Station
from ..waveform.Waveform import Waveform
from ..xcorr_utils import parameter_init


class Instrument(object):
    #Station station
    #Waveform z_component
    
    def __init__(self, Station):
        self._station = Station
        self._waveforms = {}

    def set_filters(self, filters):
        self._filters = filters
    
    def push_waveform(self, path, component, envsmooth = 1500, env_exp = 1.5, min_weight = 0.1, 
                taper_length = 1000, plot = False, normalization = None, apply_broadband_filter = True,
                broadband_filter = [200,1], filter_order = 4):
        #print "path", path
        waveform = Waveform(path)
        if (len(self._waveforms) == 0):
            lat,lon,elev = waveform.get_coordinates()
            #print type(self._station)
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

    def get_starttime(self,component,i):
        return self._waveforms[component][i].get_starttime()

    def print_waveforms(self, component, extended = False):
        if (len(self._waveforms) > 0):
            i = 0
            while i < len(self._waveforms[component]):
                self._waveforms[component][i].print_waveform(extended)
                i += 1
        else:
            print "Empty container"

