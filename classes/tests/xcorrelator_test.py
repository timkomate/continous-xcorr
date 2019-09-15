from ..xcorrealtor.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset

from timeit import default_timer as timer

start = timer()

#data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])
#data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])#data = Dataset("/gaussdata/Seismologie/PannonianBasin/data/2017",["HHZ"])
data = Dataset("/gaussdata/Seismologie/PannonianBasin/data",["HHZ"])
#data.read_dataset()
#data.save_json("../test_py")
data.load_json("../test_py")
intersect =  data.intersect("HHZ","HU","BUD","HHZ","Z3","A263A")

xc = Xcorrelator("HHZ","HU","BUD","HHZ","Z3","A263A", intersect)
xc.read_waveforms()
xc.correct_waveform_lengths()
xc.xcorr(600)
end = timer()
print(end - start)  
#xc.fft()
