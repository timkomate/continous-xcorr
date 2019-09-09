from ..xcorrealtor.Xcorrelator import Xcorrelator
from ..dataset.Dataset import Dataset

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"])

#data.save_json("../test_py")
data.load_json("../test_py")
intersect =  data.intersect("HHZ","HU","ABAH","HHZ","Z3","A263A")

xc = Xcorrelator("HHZ","HU","ABAH","HHZ","Z3","A263A", intersect)
xc.read_waveforms()
xc.correct_waveform_lengths()
xc.xcorr(6000)
#xc.fft()