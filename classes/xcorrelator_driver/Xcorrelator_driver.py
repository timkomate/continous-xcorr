from ..xcorrelator.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils.setup_logger import logger
from ..xcorr_utils import parameter_init
import multiprocessing
from timeit import default_timer as timer

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
        f = open(input_name, 'r')
        self._file = f.read().splitlines()
        self._c = len(self._file)
        for line in self._file:
            start = timer()
            network1, station1, component1 = line.split(' ')[0].split(".")
            network2, station2, component2 = line.split(' ')[1].split(".")
            intersect =  self._dataset.intersect(
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
                json_path = "./stations.json"
            )
            xc.read_waveforms(
                filters = parameter_init.filters,
                envsmooth = parameter_init.envsmooth,
                env_exp =  parameter_init.env_exp,
                min_weight = parameter_init.min_weight,
                taper_length = parameter_init.taper_lenght_timedomain,
                plot = parameter_init.verbose
            )
            xc.correct_waveform_lengths()
            xc.xcorr(
                maxlag = parameter_init.maxlag, 
                spectrumexp= parameter_init.spectrumexp,
                espwhitening = parameter_init.espwhitening,
                taper_length = parameter_init.taper_length_whitening,
                verbose =  parameter_init.verbose
            )
            save_path = xc.save_ccf(
                path = parameter_init.save_path,
                extended_save = parameter_init.extended_save
            )
            #end = timer()
            logger.debug("%s.%s-%s.%s::%s" % (network1,station1,network2,station2, timer() - start))
            del xc # is this really necessarily?

