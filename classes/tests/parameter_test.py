from ..xcorrelator.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils import parameter_init
from ..xcorr_utils.setup_logger import logger
import numpy as np
import glob
import math
import pandas as pd
from timeit import default_timer as timer

start = timer()

data = Dataset(
    path = parameter_init.dataset_path,
    components = parameter_init.components,
    years = parameter_init.years
)

if(parameter_init.build_dataset):
    data.read_dataset(
        file_type = parameter_init.file_type
    )
    data.save_json(parameter_init.dataset_name)


data.load_json(parameter_init.dataset_name)
input_path = parameter_init.input_path
input_list = [f for f in glob.glob("{}/*.text*".format(input_path))]
print input_list

for envsmooth in np.arange(500,4000,100):
    parameter_init.envsmooth = envsmooth
    print parameter_init.envsmooth
    for file in input_list:
        print file
        df = pd.read_csv(
            filepath_or_buffer = file, 
            delimiter= " ",
            header= None,
            comment= "#"
        )
        df.columns = ["network1", "station1", "component1", "network2", "station2", "component2"]
        for index, row in df.iterrows():
            start = timer()
            network1 = row["network1"]
            station1 = row["station1"]
            component1 = row["component1"]
            network2 = row["network2"]
            station2 = row["station2"]
            component2 = row["component2"]
            intersect =  data.intersect(
                component1 = component1,
                network1 = network1,
                station1 = station1,
                component2 = component2,
                network2 = network2,
                station2 = station2
            )
            xc = Xcorrelator(
                component1 = component1,
                network1 = network1,
                station1 = station1,
                component2 = component2,
                network2 = network2,
                station2 = station2,
                paths = intersect,
            )
            if (parameter_init.max_waveforms > 0):
                t = math.ceil(float(len(intersect))/parameter_init.max_waveforms)
            else:
                t = 1
            
            for i in range(int(t)):
                xc.read_waveforms(
                    max_waveforms = parameter_init.max_waveforms,
                    maxlag = parameter_init.maxlag,
                    filters = parameter_init.filters,
                    filter_order = parameter_init.filter_order_tdn,
                    envsmooth = parameter_init.envsmooth,
                    env_exp = parameter_init.env_exp,
                    min_weight = parameter_init.min_weight,
                    taper_length = parameter_init.taper_length_timedomain,
                    plot = parameter_init.plot,
                    apply_broadband_filter_tdn= parameter_init.apply_broadband_filter_tdn,
                    broadband_filter_tdn = parameter_init.broadband_filter_tdn
                )  
                xc.correct_waveform_lengths()
                xc.xcorr(
                    maxlag = parameter_init.maxlag,
                    spectrumexp = parameter_init.spectrumexp,
                    espwhitening = parameter_init.espwhitening,
                    taper_length_whitening = parameter_init.taper_length_whitening,
                    verbose = parameter_init.plot,
                    apply_broadband_filter = parameter_init.apply_broadband_filter_whitening,
                    broadband_filter = parameter_init.broadband_filter_whitening,
                    apply_spectral_whitening = parameter_init.apply_spectral_whitening,
                    filter_order = parameter_init.filter_order_whitening
                )
            xc.calculate_linear_stack()
            save_path = xc.save_ccf(
                path = parameter_init.save_path, 
                extended_save= True,
                tested_parameter = "_envsmooth{}".format(envsmooth)
            )
            logger.info("{}.{}.{}-{}.{}.{}::{}".format(
                network1,station1,component1, 
                network2,station2,component2, 
                timer() - start)
            )