from ..Xcorrelator_driver.Xcorrelator_driver import Xcorrelator_driver
import multiprocessing
from ..dataset.Dataset import Dataset  

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ"], [2017, 2018])
data.load_json("./dataset.json")

xc_d = Xcorrelator_driver(data)
p = multiprocessing.Pool(2)
p.map(xc_d.run,["./xaa", "./xab",  "./xac"])