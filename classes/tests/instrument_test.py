from ..waveform.Waveform import Waveform
from ..instrument.Instrument import Instrument
from ..station.Station import Station

sta = Station("HU", "BUD", None, None, None)

path = "/home/mate/PhD/codes/continous_waveform_downloader/test/5HZ/Z/2017/20170101/HU.BUD.Z_VELtdn_2017-01-01.mat"
wf = Waveform(path)

inst = Instrument(sta)
inst.get_station().info()
inst.push_waveform(
    path = path, 
    component = "Z"
)
inst.push_waveform(
    path = path, 
    component = "Z"
)
#inst.print_station()
inst.print_waveforms(
    component = "Z"
)

waveform = inst.get_waveform(
    component = "Z",
    i = 0
)
waveform.print_waveform()

inst.clear()

inst.print_waveforms(
    component = "Z"
)

inst.get_station().info()