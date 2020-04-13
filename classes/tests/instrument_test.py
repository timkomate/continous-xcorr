from ..waveform.Waveform import Waveform
from ..instrument.Instrument import Instrument
from ..station.Station import Station

sta = Station("HU", "BUD", None, None, None)

path1 = "/home/mate/PhD/codes/continous_waveform_downloader/test/5HZ/Z/2017/20170101/HU.BUD.Z_VELtdn_2017-01-01.mat"
path2 = "/home/mate/PhD/codes/continous_waveform_downloader/test/5HZ/Z/2017/20170101/HU.PSZ.Z_VELtdn_2017-01-01.mat"


inst = Instrument(sta)
inst.get_station().info()
print 
inst.push_waveform(
    path = path1, 
    component = "Z"
)
inst.push_waveform(
    path = path2, 
    component = "Z"
)
#inst.print_station()
"""inst.print_waveforms(
    component = "Z"
)"""

"""inst.get_waveforms_mtx(
    component = "Z"
)"""

"""waveform = inst.get_waveform(
    component = "Z",
    i = 0
)
waveform.print_waveform()

inst.clear()

inst.print_waveforms(
    component = "Z"
)

inst.get_station().info()"""