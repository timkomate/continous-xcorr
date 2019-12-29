from ..acorrelator.Acorrelator import Acorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils import parameter_init
from ..xcorr_utils.setup_logger import logger
import numpy as np
import glob
import math
import pandas as pd
from timeit import default_timer as timer


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

data.read_dataset(
    file_type= "_VEL_"
)

data.load_json(parameter_init.dataset_name)
input_path = parameter_init.input_path
input_list = [f for f in glob.glob("%s/*.text*" % (input_path))]
print input_list

for file in input_list:
    print file
    df = pd.read_csv(
        filepath_or_buffer = file, 
        delimiter= ".",
        header= None,
        comment= "#"
    )
    df.columns = ["network", "station", "component"]
    for index, row in df.iterrows():
        start = timer()
        network = row["network"]
        station = row["station"]
        component = row["component"]
        intersect =  data.get_folders(
            component = component,
            network = network,
            station = station
        )

        ac = Acorrelator(
            component = component,
            network = network,
            station = station,
            paths = intersect,
        )

        t = math.ceil(float(len(intersect))/400)

        print t
        for i in range(int(t)):
            ac.read_waveforms(
                max_waveforms = parameter_init.max_waveforms,
                maxlag = parameter_init.maxlag,
                filters = parameter_init.filters,
                filter_order = parameter_init.filter_order_tdn,
                envsmooth = parameter_init.envsmooth,
                env_exp = parameter_init.env_exp,
                min_weight = parameter_init.min_weight,
                taper_length = parameter_init.taper_lenght_timedomain,
                plot = parameter_init.plot,
                apply_broadband_filter_tdn= parameter_init.apply_broadband_filter_tdn,
                broadband_filter_tdn = parameter_init.broadband_filter_tdn
            )   
            ac.acorr(
                maxlag = parameter_init.maxlag,
                spectrumexp = parameter_init.spectrumexp,
                espwhitening = parameter_init.espwhitening,
                taper_length_whitening = parameter_init.taper_length_whitening,
                verbose = parameter_init.plot,
                apply_broadband_filter = parameter_init.apply_broadband_filter_whitening,
                broadband_filter = parameter_init.broadband_filter_whitening,
                filter_order = parameter_init.filter_order_whitening
            )
        save_path = ac.save_acf("./", extended_save= True)
        logger.debug("{}.{}.{}::{}".format(network,station, component, timer() - start))