import collections
import os
import json

class Dataset(object):
#private:
    #collection data
    #string[] matfiles
    def __init__(self,path,components):
        self._dataset = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
        self._path = path
        self._components = components

    def read_dataset(self):
        for component in self._components:
            root_dir = "%s/%s/2017/" % (self._path, component)
            for dir_name, subdir_list, file_list in os.walk(root_dir):
                    for fname in file_list:
                        #print(fname)
                        network = fname.split('.')[0]
                        station = fname.split('.')[1]
                        self.push_to_dataset(component,network,station,dir_name,fname)
    
    def print_dataset(self):
        print self._dataset

    def get_folders(self):
        return self._dataset
    
    def get_matfiles(self,component,network,station):
        return self._dataset[component][network][station]["matfiles"]

    def get_folders(self,component,network,station):
        return self._dataset[component][network][station]["folders"]

    def save_json(self,savepath):
        with open(savepath, 'w') as fp:
            json.dump(self._dataset, fp, sort_keys=True, indent=2)

    def load_json(self,loadpath):
        with open(loadpath, 'r') as fp:
            data = json.load(fp)
        self._dataset = data

    def push_to_dataset(self,component,network, station,dir_name,fname):
        fold = "folders"
        if component in self._dataset:
            if network in self._dataset[component]:
                if station in self._dataset[component][network]:
                    self._dataset[component][network][station]["folders"].append(dir_name)
                    self._dataset[component][network][station]["matfiles"].append(fname)
                    return
        #print component, network, station
        self._dataset[component][network][station]["folders"] = [dir_name]
        self._dataset[component][network][station]["matfiles"] = [fname]

    def intersect(self,component1, network1, station1,component2, network2, station2):
        lst1 = self.get_folders(component1, network1, station1)
        lst2 = self.get_folders(component2,network2,station2)
        # Use of hybrid method 
        temp = set(lst2) 
        lst3 = [value for value in lst1 if value in temp] 
        return lst3 
  
        
