from ..acorrelator.Acorrelator import Acorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init, xcorr_utils
import multiprocessing
from timeit import default_timer as timer
import pandas as pd
import math
from ..exceptions.IntersectionError import IntersectionError
from ..exceptions.FileExcist import FileExcistError

class Acorrelator_driver(object):
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
        df.columns = ["network", "station", "component"]
        for index, row in df.iterrows():
            try:
            #if True:
                reading_time = 0
                start = timer()
                network = row["network"]
                station = row["station"]
                component = row["component"]
                message = "{}.{}.{}".format(network,station,component)
                intersect =  self._dataset.get_folders(
                    component = component,
                    network = network,
                    station = station
                )
                """if (not parameter_init.overwrite):
                    excist = xcorr_utils.cf_excist(
                        savepath = parameter_init.save_path,
                        network1 = network1,
                        station1 = station1,
                        network2 = network2,
                        station2 = station2,
                        corrflag="ACF"
                    )
                    if excist:
                        fee_msg = "CCF between {}.{} and {}.{} already excist".format(network1,station1,network2,station2)
                        raise FileExcistError(fee_msg)"""

                """if (len(intersect) < parameter_init.min_days):
                    msg = "IntersectionError between {}.{} and {}.{} ({} days)".format(network1,station1, 
                        network2, station2, len(intersect))
                    raise IntersectionError(msg)"""

                ac = Acorrelator(
                    component = component,
                    network = network,
                    station = station,
                    paths = intersect,
                    file_type= parameter_init.file_type
                )
                
                if (parameter_init.max_waveforms > 0):
                    t = math.ceil(float(len(intersect))/parameter_init.max_waveforms)
                else:
                    t = 1

                for i in range(int(t)):
                    read = timer()
                    ac.read_waveforms(
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
                    ac.acorr_pcc(
                        maxlag = parameter_init.maxlag,
                        spectrumexp = parameter_init.spectrumexp,
                        espwhitening = parameter_init.espwhitening,
                        taper_length_whitening = parameter_init.taper_length_whitening,
                        verbose = parameter_init.plot,
                        apply_broadband_filter = parameter_init.apply_broadband_filter_whitening,
                        broadband_filter = parameter_init.broadband_filter_whitening,
                        filter_order = parameter_init.filter_order_whitening
                    )
            except IntersectionError as e:
                logger.info("{}::{}::{}::{}".format(message, 0, timer() - start, -1))
                continue
            except ValueError as e:
                logger.info("{}::{}::{}::{}".format(message, 0, timer() - start, -2))
                continue
            except IOError as e:
                logger.info("{}::{}::{}::{}".format(message, 0, timer() - start, -3))
                continue
            except KeyError as e:
                logger.info("{}::{}::{}::{}".format(message, 0, timer() - start, -4))
                continue
            except FileExcistError as e:
                logger.info("{}::{}::{}::{}".format(message, 0, timer() - start, -5))
                continue
            
            ac.calculate_linear_stack()
            save_path = ac.save_acf(
                path = parameter_init.save_path,
                extended_save = parameter_init.extended_save
            )
            logger.info("{}::{}::{}::{}".format(message, ac.get_nstack(), timer() - start, reading_time))
            del ac # is this really necessarily?