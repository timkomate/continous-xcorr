#!/usr/bin/python
import sys
#sys.path.insert(0, './classes/station/')
from ..station.Station import Station
#sys.path.insert(0, './classes/waveform/')
from ..waveform.Waveform import Waveform


class Instrument(object):
    #Station station
    #Waveform z_component
    
    def __init__(self, Station, filters = []):
        self._station = Station
        self._waveforms = []
        self._filters = filters

    def set_filters(self, filters):
        self._filters = filters
    
    def push_waveform(self, path):
        waveform = Waveform(path)
        if len(self._filters):
            waveform.running_absolute_mean(filters = self._filters)
        else:
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

