#!/usr/bin/python
from classes.station.Station import Station
from classes.instrument.Instrument import Instrument
from classes.xcorrelator.Xcorrelator import Xcorrelator
from classes.dataset.Dataset import Dataset
from classes.xcorrelator_driver.Xcorrelator_driver import Xcorrelator_driver
from classes.xcorr_utils.setup_logger import logger

import numpy as np
from timeit import default_timer as timer
import multiprocessing
import os
import glob
import ConfigParser

if __name__== "__main__":
    config = ConfigParser.ConfigParser()
    config.read("./config.cfg")
    start = timer()
    data = Dataset(
        path = config.get("DEFAULT", "dataset_path"),
        components = config.get("DEFAULT", "components").split(','),
        years = config.get("DEFAULT", "years").split(',')
    )
    if (config.getboolean("DEFAULT", "build_dataset")):
        data.read_dataset()
        data.save_json(config.get("DEFAULT", "dataset_name"))
        
    data.load_json(config.get("DEFAULT", "dataset_name"))
    input_path = config.get("DEFAULT", "input_path")
    input_list = [f for f in glob.glob("%s/*.text*" % (input_path))]
    print input_list
    xc_d = Xcorrelator_driver(
        dataset = data,
        filenames = input_list,
        config_file = config
    )
    xc_d.run(
        core_number = config.getint("DEFAULT", "number_of_cpus")
    )
    end = timer()
    print "Script finished:", end - start, "seconds"
