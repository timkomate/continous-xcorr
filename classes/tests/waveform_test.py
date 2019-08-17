from ..waveform.Waveform import Waveform
import numpy as np

#/home/mate/PhD/codes/continous-xcorr/test_dataset
path = "/home/mate/PhD/codes/continous-xcorr/test_dataset/HHZ/201711140000/Z3.A263A.HHZ_VEL_2017-11-14.00-00-00.mat"
wf = Waveform(path)
print wf.get_data()
print np.shape(wf.get_data())
wf.print_waveform()