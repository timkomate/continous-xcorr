from ..xcorrelator.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset
import multiprocessing
from ..xcorr_utils.setup_logger import logger

class Xcorrelator_driver(object):
    def __init__(self, dataset, filenames):
        self.add_dataset(dataset)
        self._filenames = filenames
        logger.info("Xcorrelator")

    def add_dataset(self, dataset):
        self._dataset = dataset

    def __call__(self, filename):
        self.go(filename)

    def run(self, core_number = multiprocessing.cpu_count()):
        print "Number of cores:", core_number
        pool = multiprocessing.Pool(core_number)
        pool.map(self, self._filenames)
        pool.close()

    def go(self, input_name):
        f = open(input_name, 'r')
        self._file = f.read().splitlines()
        self._c = len(self._file)
        #filters = [[100,10],[10,5],[5,1]]
        filters = []
        for line in self._file:
            network1, station1, component1 = line.split(' ')[0].split(".")
            network2, station2, component2 = line.split(' ')[1].split(".")
            intersect =  self._dataset.intersect(component1, network1, station1, component2, network2, station2)

            xc = Xcorrelator(component1, network1, station1, component2, network2, station2, intersect, "./stations.json")
            xc.read_waveforms(filters = filters)
            xc.correct_waveform_lengths()
            xc.xcorr(
                600,
                spectrumexp= 0.7)
            xc.save_ccf("./ccfs", save_daily_ccf= False)
