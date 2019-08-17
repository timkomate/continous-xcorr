from ..station.Station import Station

sta = Station("HU", "BUD", 47, 19, 165)
sta.info()

coord = sta.get_coordinates()
print coord

netw_code = sta.get_network_code()
sta_code = sta.get_station_code()

print netw_code, sta_code
