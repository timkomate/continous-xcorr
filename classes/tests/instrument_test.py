from ..waveform.Waveform import Waveform
from ..instrument.Instrument import Instrument
from ..station.Station import Station

sta = Station("HU", "BUD", 47, 19, 165)

path = "/home/mate/PhD/codes/continous-xcorr/test_dataset/HHZ/201711140000/Z3.A263A.HHZ_VEL_2017-11-14.00-00-00.mat"
wf = Waveform(path)

inst = Instrument(Station)
inst.push_waveform(wf)
inst.push_waveform(wf)
#inst.print_station()
inst.print_waveforms()

waveform = inst.get_waveform(0)
waveform.print_waveform()