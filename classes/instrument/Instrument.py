#!/usr/bin/python
import sys
sys.path.insert(0, './classes/station/')
from Station import Station
sys.path.insert(0, './classes/waveform/')
from Waveform import Waveform


class Instrument(object):
    #Station station
    #Waveform z_component
    
    def __init__(self, Station):
        self._station = Station
        self._waveforms = []
    
    def push_waveform(self, path):
        waveform = Waveform(path)
        waveform.binary_normalization()
        self._waveforms.append(waveform)

    def get_sampling_rate(self):
        return self._waveforms[0].get_sampling_rate()
    
    #def print_station(self):
    #    self._station.info()

    def get_waveform(self, i):
        return self._waveforms[i]

    def print_waveforms(self, extended = False):
        i = 0
        while i < len(self._waveforms):
            self._waveforms[i].print_waveform(extended)
            i += 1

