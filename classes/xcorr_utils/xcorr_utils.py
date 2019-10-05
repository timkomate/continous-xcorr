import numpy as np
from scipy import signal, fftpack, io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math

def calc_distance_deg(s_coordinates, e_coordinates):
    slat = s_coordinates[0]
    slon = s_coordinates[1]
    elat = e_coordinates[0]
    elon = e_coordinates[1]
    FLATTENING = 0.00335281066474
    f = (1. - FLATTENING) * (1. - FLATTENING)
    geoc_elat = math.atan(f * math.tan((math.pi/180) * elat))
    celat = math.cos(geoc_elat)
    selat = math.sin(geoc_elat)
    geoc_slat = math.atan(f * math.tan((math.pi/180) * slat))
    rdlon = (math.pi/180) * (elon - slon)
    cslat = math.cos(geoc_slat)
    sslat = math.sin(geoc_slat)
    cdlon = math.cos(rdlon)
    sdlon = math.sin(rdlon)
    cdel = sslat * selat + cslat * celat * cdlon
    cdel if cdel<1 else 1
    cdel if cdel>-1 else -1
    delta = (180/math.pi) * math.acos(cdel)
    return delta
    

def calc_distance_km(s_coordinates, e_coordinates):
    slat = math.radians(s_coordinates[0])
    slon = math.radians(s_coordinates[1])
    elat = math.radians(e_coordinates[0])
    elon = math.radians(e_coordinates[1])
    R = 6371.0
    dlon = slon - elon
    dlat = slat - elat
    a = math.sin(dlat / 2)**2 + math.cos(elat) * math.cos(slat) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c
    
def load_station_infos(json_path):
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    return data
    
def find_station_coordinates(data, network, station):
    elev = data[network][station]["elevation"]
    lat = data[network][station]["latitude"]
    lon = data[network][station]["longitude"]
    return [lat, lon, elev]

def downweight_ends(data, wlength):
    w = (1 - np.cos((math.pi / wlength) * (np.arange(0,wlength,1) + 1)))/2
    data[0:int(wlength)] = data[0:int(wlength)]*w
    w = np.flipud(w)
    data[-int(wlength):] = data[-int(wlength):]*w
    return data