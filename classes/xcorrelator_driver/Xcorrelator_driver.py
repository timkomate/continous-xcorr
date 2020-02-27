from ..xcorrelator.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init
import multiprocessing
from timeit import default_timer as timer
import pandas as pd
import math
from ..exceptions.IntersectionError import IntersectionError

class Xcorrelator_driver(object):
    def __init__(self, dataset, filenames):
        self.add_dataset(dataset)
        self._filenames = filenames
    
    def add_dataset(self, dataset):
        self._dataset = dataset
    
    def __call__(self, filename):
        self.go(filename)

    def run(self, core_number = multiprocessing.cpu_count()):
        if core_number < 1:
            core_number = multiprocessing.cpu_count()
        print "Number of cores:", core_number
        pool = multiprocessing.Pool(core_number)
        pool.map(self, self._filenames)
        pool.close()
        pool.join()

    def go(self, input_name):
        print input_name
        df = pd.read_csv(
            filepath_or_buffer = input_name, 
            delimiter= " ",
            header= None,
            comment= "#"
        )
        df.columns = ["network1", "station1", "component1", "network2", "station2", "component2"]
        for index, row in df.iterrows():
            try:
                reading_time = 0
                start = timer()
                network1 = row["network1"]
                station1 = row["station1"]
                component1 = row["component1"]
                network2 = row["network2"]
                station2 = row["station2"]
                component2 = row["component2"]
                message = "{}.{}.{}-{}.{}.{}".format(network1,station1,component1,network2,station2,component2)
                intersect =  self._dataset.intersect(
                    component1 = component1,
                    network1 = network1,
                    station1 = station1,
                    component2 = component2,
                    network2 = network2,
                    station2 = station2
                )

                if (len(intersect) == 0):
                    msg = "IntersectionError between {}.{} and {}.{} ({} days)".format(network1,station1, 
                        network2, station2, len(intersect))
                    raise IntersectionError(msg)

                xc = Xcorrelator(
                    component1 = component1,
                    network1 = network1,
                    station1 = station1,
                    component2 = component2,
                    network2 = network2,
                    station2 = station2,
                    paths = intersect,
                    file_type= parameter_init.file_type
                )
                
                if (parameter_init.max_waveforms > 0):
                    t = math.ceil(float(len(intersect))/parameter_init.max_waveforms)
                else:
                    t = 1

                for i in range(int(t)):
                    read = timer()
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
                    reading_time += timer() - read

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
            except IntersectionError as e:
                logger.info("{}::{}::{}::{}".format(message, xc.get_nstack(), timer() - start, -1))
                del xc
                continue
            except ValueError as e:
                logger.info("{}::{}::{}::{}".format(message, xc.get_nstack(), timer() - start, -2))
                del xc
                continue
            
            xc.calculate_linear_stack()
            save_path = xc.save_ccf(
                path = parameter_init.save_path,
                extended_save = parameter_init.extended_save
            )
            logger.info("{}::{}::{}::{}".format(message, xc.get_nstack(), timer() - start, reading_time))
            del xc # is this really necessarily?