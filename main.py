#!/usr/bin/python
from classes.station.Station import Station
from classes.instrument.Instrument import Instrument
from classes.xcorrealtor.Xcorrelator import Xcorrelator
from classes.dataset.Dataset import Dataset
import numpy as np
from timeit import default_timer as timer

def main():
    start = timer()
    filters = [[100,10],[10,5],[5,1]]
    filters = []
    #data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])
    data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"], [2017, 2018])
    #data = Dataset("/gaussdata/Seismologie/PannonianBasin/data",["HHZ"])
    #data = Dataset("/media/timko/Maxtor/",["HHZ"])
    #data = Dataset("/maxwelldata/pannonian/PannonianBasin/data",["HHZ"])

    data.read_dataset()
    data.save_json("./dataset.json")
    data.load_json("./dataset.json")
    intersect =  data.intersect("HHZ","Z3","A263A","HHZ","HU","ABAH")

    xc = Xcorrelator("HHZ","Z3","A263A","HHZ","HU","ABAH", intersect, "./stations.json")
    xc.read_waveforms(filters= filters)
    xc.correct_waveform_lengths()
    #for i in np.arange(0.1,2.5,0.1):
    xc.xcorr(600, spectrumexp= 0.7)
#xc.save_figures("./figures/")
    end = timer()
    print "Script finished:", end - start
    xc.save_ccf("./ccfs")
#xc.fft()

  
if __name__== "__main__":
    main()
