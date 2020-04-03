#!/usr/bin/python
from classes.station.Station import Station
from classes.instrument.Instrument import Instrument
from classes.xcorrelator.Xcorrelator import Xcorrelator
from classes.acorrelator.Acorrelator import Acorrelator
from classes.dataset.Dataset import Dataset
from classes.xcorrelator_driver.Xcorrelator_driver import Xcorrelator_driver
from classes.acorrelator_driver.Acorrelator_driver import Acorrelator_driver
from classes.xcorr_utils.setup_logger import logger
from classes.xcorr_utils import parameter_init
import numpy as np
from timeit import default_timer as timer
import multiprocessing
import os
import glob

if __name__== "__main__":
    start = timer()
    data = Dataset(
        path = parameter_init.dataset_path,
        components = parameter_init.components,
        years = parameter_init.years
    )
    if (parameter_init.build_dataset):
        data.read_dataset(
            file_type = parameter_init.file_type
        )
        data.save_json(parameter_init.dataset_name)
        
    print parameter_init.dataset_name
    data.load_json(parameter_init.dataset_name)
    input_path = parameter_init.input_path
    input_list = [f for f in glob.glob("%s/*.text*" % (input_path))]
    print input_list
    ac_d = Acorrelator_driver(
        dataset = data,
        filenames = input_list,
    )
    ac_d.run(
        core_number = parameter_init.number_of_cpus
    )
    end = timer()
    print "Script finished:", end - start, "seconds"
