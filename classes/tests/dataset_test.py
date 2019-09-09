from ..dataset.Dataset import Dataset

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ","HHE"])
data.read_dataset()
data.save_json("../test_py")
print data.intersect("HHZ","HU","ABAH","HHZ","Z3","A263A")

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ","HHE"])
data.load_json("../test_py")
print data.intersect("HHZ","HU","ABAH","HHZ","Z3","A263A")