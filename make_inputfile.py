#/usr/bin/python
import os
import itertools

def get_file_infos(filename):
    splitted_filename = filename.split('.')
    network = splitted_filename[0]
    station = splitted_filename[1]
    component = splitted_filename[2].split('_')[0]
    return [network, station, component]

path = "/mnt/storage_A/mate/ambient_noise_data/"
component = "Z/"
year = "2017"


data_directory = path + component + year
output_name = "./combinations{}.text".format(year)
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
        entry = "{} {} {}".format(network, station, component)
        #print entry
        if entry not in station_list:
            station_list.append(entry)
            print entry

a = list(itertools.combinations(station_list, 2))
for i in a:
    output.write("{} {}\n".format(i[0], i[1]))
