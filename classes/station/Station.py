class Station(object):
    def __init__(self, network_code, station_code, lat, lon, elev):
        self._network_code = network_code
        self._station_code = station_code
        self._lat = lat
        self._lon = lon
        self._elev = elev

    def info(self):
        print "Network code:", self._network_code
        print "Station code:", self._station_code
        print "Latitude:", self._lat
        print "Longitude:", self._lon
        print "Elevation:", self._elev

    def set_coordinates(self, lat, lon, elev):
        self._lat = lat
        self._lon = lon
        self._elev = elev
    
    def get_network_code(self):
        return self._network_code

    def get_station_code(self):
        return self._station_code
    
    def get_coordinates(self):
        return [self._lat, self._lon, self._elev]
    