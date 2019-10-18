from ..xcorrelator.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset
from ..xcorr_utils.setup_logger import logger
import multiprocessing
from timeit import default_timer as timer

class Xcorrelator_driver(object):
    def __init__(self, dataset, filenames, config_file):
        self.add_dataset(dataset)
        self._filenames = filenames
        self._config_file = config_file
    
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
        filters = [[100,10],[10,5],[5,1]]
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
                filters = filters,
                envsmooth = self._config_file.getint("DEFAULT", "evnsmooth"),
                env_exp = self._config_file.getfloat("DEFAULT", "env_exp"),
                min_weight = self._config_file.getfloat("DEFAULT", "min_weight"),
                taper_length = self._config_file.getint("DEFAULT", "taper_lenght_timedomain"),
                plot = self._config_file.getboolean("DEFAULT", "verbose")
            )
            xc.correct_waveform_lengths()
            xc.xcorr(
                maxlag = self._config_file.getint("DEFAULT", "maxlag"), 
                spectrumexp= self._config_file.getfloat("DEFAULT", "spectrumexp"),
                espwhitening = self._config_file.getfloat("DEFAULT", "espwhitening"),
                taper_length = self._config_file.getint("DEFAULT", "taper_length_whitening"),
                verbose = self._config_file.getboolean("DEFAULT", "verbose")
            )
            save_path = xc.save_ccf(
                path = self._config_file.get("DEFAULT", "save_path"),
                extended_save = self._config_file.getboolean("DEFAULT", "extended_save")
            )
            end = timer()
            logger.debug("%s.%s-%s.%s::%s" % (network1,station1,network2,station2, end - start))
            del xc # is this really necessarily?

