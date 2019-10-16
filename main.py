#!/usr/bin/python
from classes.station.Station import Station
from classes.instrument.Instrument import Instrument
from classes.xcorrelator.Xcorrelator import Xcorrelator
from classes.dataset.Dataset import Dataset
import numpy as np
from timeit import default_timer as timer
from classes.xcorrelator_driver.Xcorrelator_driver import Xcorrelator_driver
import multiprocessing
import os
from classes.xcorr_utils.setup_logger import logger

def main1():
    start = timer()
    filters = [[100,10],[10,5],[5,1]]
    filters = []
    #data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])
    #data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"], [2017, 2018])
    #data = Dataset("/gaussdata/Seismologie/PannonianBasin/data",["HHZ"])
    data = Dataset("/media/timko/Maxtor/",["HHZ"], [2017, 2018])
    #data = Dataset("/maxwelldata/pannonian/PannonianBasin/data",["HHZ"])

    data.read_dataset()
    data.save_json("./dataset.json")
    data.load_json("./dataset.json")
    intersect =  data.intersect("HHZ","Z3","A263A","HHZ","HU","BUD")

    xc = Xcorrelator("HHZ","Z3","A263A","HHZ","HU","BUD", intersect, "./stations.json")
    xc.read_waveforms(filters= filters)
    xc.correct_waveform_lengths()
    xc.xcorr(600, spectrumexp= 0.7)
    end = timer()
    print "Script finished:", end - start
    xc.save_ccf("./ccfs", save_daily_ccf= False)


if __name__== "__main__":
    #data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"], [2017, 2018])
    #data.load_json("./dataset.json")
    data = Dataset("/media/timko/Maxtor/",["HHZ"], [2017, 2018])
    data.load_json("./dataset.json")
    logger.debug("Messzidzs")
    xc_d = Xcorrelator_driver(data,["./xcorr_stations.text_aa", "./xcorr_stations.text_ab",  "./xcorr_stations.text_ac"])
    xc_d.run(core_number = 2)
