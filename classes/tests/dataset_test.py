from ..dataset.Dataset import Dataset

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ","HHE"], [2017, 2018])
data.read_dataset()
data.save_json("./dataset.json")
print data.intersect("HHZ","HU","ABAH","HHZ","Z3","A263A")

data = Dataset("/home/mate/PhD/codes/continous-xcorr/test_dataset",["HHZ","HHE"], [2017])
data.load_json("./dataset.json")
print data.intersect("HHZ","HU","ABAH","HHZ","Z3","A263A")