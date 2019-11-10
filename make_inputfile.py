#/usr/bin/python
import os
import itertools

def get_file_infos(filename):
    splitted_filename = filename.split('.')
    network = splitted_filename[0]
    station = splitted_filename[1]
    component = splitted_filename[2].split('_')[0]
    return [network, station, component]

path = "/home/mate/PhD/codes/continous_waveform_downloader/test/"
component = "Z"

data_directory = path + component
output_name = "./combinations.text"
output = open(output_name, "w")

ll = os.walk(data_directory)
station_list = []
while True:
    try:
        pwd, sub_dirs, files = ll.next()
    except StopIteration:
        break
    for file in files:
        #print "%s/%s" % (pwd, file)
        network, station, component = get_file_infos(file)
        entry = "%s.%s.%s" % (network, station, component)
        #print entry
        if entry not in station_list:
            station_list.append(entry)
            print entry

a = list(itertools.combinations(station_list, 2))
for i in a:
    output.write("%s %s\n" % (i[0], i[1]))