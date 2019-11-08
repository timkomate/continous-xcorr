#!/usr/bin/python
import sys
from ..station.Station import Station
from ..waveform.Waveform import Waveform
from ..xcorr_utils import parameter_init


class Instrument(object):
    #Station station
    #Waveform z_component
    
    def __init__(self, Station, filters = []):
        self._station = Station
        self._waveforms = []
        self._filters = filters

    def set_filters(self, filters):
        self._filters = filters
    
    def push_waveform(self, path, envsmooth = 1500, env_exp = 1.5, min_weight = 0.1, 
                taper_length = 1000, plot = False):
        waveform = Waveform(path)
        if (parameter_init.running_absolute_mean_normalization):
            print "ramn"
            waveform.running_absolute_mean(
                filters = self._filters,
                envsmooth = envsmooth,
                env_exp = env_exp,
                min_weight = min_weight, 
                taper_length = taper_length, 
                plot = False
            )
        elif (parameter_init.binary_normalization):
            print "bn"
            waveform.binary_normalization()
        self._waveforms.append(waveform)

    def get_sampling_rate(self):
        return self._waveforms[0].get_sampling_rate()

    def clear(self):
        self._waveforms = []
    
    def get_station_coordinates(self):
        return self._station.get_station_coordinates()

    def get_station_code(self):
        return self._station.get_station_code()

    def get_waveform(self, i):
        return self._waveforms[i]

    def print_waveforms(self, extended = False):
        i = 0
        while i < len(self._waveforms):
            self._waveforms[i].print_waveform(extended)
            i += 1

