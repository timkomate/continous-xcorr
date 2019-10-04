from ..xcorrealtor.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset

from timeit import default_timer as timer

start = timer()

filters = [[100,10],[10,5],[5,1]]
#filters = []
#data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])
#data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])#data = Dataset("/gaussdata/Seismologie/PannonianBasin/data/2017",["HHZ"])
#data = Dataset("/gaussdata/Seismologie/PannonianBasin/data",["HHZ"])
data = Dataset("/media/timko/Maxtor/",["HHZ"])

#data = Dataset("/maxwelldata/pannonian/PannonianBasin/data",["HHZ"])

data.read_dataset()
data.save_json("../test_py")
data.load_json("../test_py")
intersect =  data.intersect("HHZ","Z3","A263A","HHZ","HU","BUD")

xc = Xcorrelator("HHZ","Z3","A263A","HHZ","HU","BUD", intersect, "./stations.json")
xc.read_waveforms(filters= filters)
xc.correct_waveform_lengths()
xc.xcorr(600)
#xc.save_figures("./figures/")
end = timer()
print(end - start)
xc.save_ccf("./save_running")
#xc.fft()
