import numpy as np
from scipy import signal, fftpack, io
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import math
import glob

def calc_distance_deg(s_coordinates, e_coordinates, FLATTENING = 0.00335281066474):
    slat = s_coordinates[0]
    slon = s_coordinates[1]
    elat = e_coordinates[0]
    elon = e_coordinates[1]
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
    

def calc_distance_km(s_coordinates, e_coordinates, R = 6371.0):
    slat = math.radians(s_coordinates[0])
    slon = math.radians(s_coordinates[1])
    elat = math.radians(e_coordinates[0])
    elon = math.radians(e_coordinates[1])
    dlon = slon - elon
    dlat = slat - elat
    a = math.sin(dlat / 2)**2 + math.cos(elat) * math.cos(slat) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def angular_distance(s_coordinates, e_coordinates, R = 6371.0):
    slat = math.radians(s_coordinates[0])
    slon = math.radians(s_coordinates[1])
    elat = math.radians(e_coordinates[0])
    elon = math.radians(e_coordinates[1])
    dlon = slon - elon
    dlat = slat - elat
    a = math.sin(dlat / 2)**2 + math.cos(elat) * math.cos(slat) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return c
    
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

def nextpow2(x):
    return 1<<(x-1).bit_length()

def spectral_whitening( data, sampling_rate, spectrumexp = 0.7, 
                        espwhitening = 0.05, taper_length = 100, 
                        apply_broadband_filter = True, broadband_filter = [200,1], 
                        filter_order = 4, plot = False):
    if (plot):
        plt.plot(data)
        plt.title("original dataset")
        plt.show()
        
    data = signal.detrend(
        data = data,
        type="linear"
    )
    spectrum = np.fft.rfft(
        a = data,
        n = nextpow2(len(data))
    )
    spectrum_abs = np.abs(spectrum)
    if (plot):
        f = np.fft.rfftfreq(nextpow2(len(data)), d=1./sampling_rate)
        plt.plot(f,spectrum)
        plt.plot(f,spectrum_abs)
        plt.title("specrtum and ampl. spectrum")
        plt.show()
        
    water_level = np.mean(spectrum_abs) * espwhitening
    spectrum_abs[(spectrum_abs < water_level)] = water_level
        
    if (plot):
        plt.plot(f,spectrum_abs)
        plt.title("spectrum after water level")
        plt.show()
        fig, axs = plt.subplots(3)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(spectrum)
        axs[1].plot(np.power(spectrum_abs,spectrumexp))
        axs[2].plot(spectrum_abs)
        plt.show()
        
    #whitening
    spectrum = np.divide(spectrum, np.power(spectrum_abs,spectrumexp))
    #spectrum = downweight_ends(spectrum, wlength = (taper_length * sampling_rate))
    #spectrum[0] = 0

    if (plot):
        plt.plot(f,np.abs(spectrum))
        plt.title("spectrum after whitening")
        plt.show()

    whitened = np.fft.irfft(
        a = spectrum,
        n = nextpow2(len(data))
    )
    whitened = whitened[0:len(data)]
        
    whitened = signal.detrend(
        data = whitened,
        type="linear"
    )
        
    whitened =  downweight_ends(
        data = whitened,
        wlength= taper_length * sampling_rate
    )
    
    if (apply_broadband_filter):
        nyf = (1./2) * sampling_rate
        [b,a] = signal.butter(
            N = filter_order,
            Wn = [(1./broadband_filter[0])/nyf,(1./broadband_filter[1])/nyf], 
            btype='bandpass'
        )
        whitened = signal.filtfilt(
            b = b,
            a = a,
            x = whitened
        )

    if (plot):
        plt.plot(whitened)
        plt.title("whitened signal after filtering")
        plt.show()
    #remove mean
    whitened = whitened - np.mean(np.abs(whitened))
    return whitened



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

def cf_excist(savepath,network1, station1, network2, station2, corrflag = "*", components = "*", nstack = "*", ccftype = "*"):
    if(glob.glob("{}/{}_{}_{}_{}_{}_{}_{}_{}.mat".format(savepath,corrflag,network1,station1,network2,station2,components,nstack,ccftype))):
        return True
    elif(glob.glob("{}/{}_{}_{}_{}_{}_{}_{}_{}.mat".format(savepath,corrflag,network2,station2,network1,station1,components,nstack,ccftype))):
        return True
    else:
        return False

import scipy.signal
import numpy as np

#===========================================================
# Routine by Luis-Fabian Bonilla (IPGP & IFSTTAR), Jan 2020.
#===========================================================

# Tapering with a Hanning window

def taper(x,p):
    if p <= 0.0:
        return x
    else:
        f0 = 0.5
        f1 = 0.5
        n  = len(x)
        nw = int(p*n)

        if nw > 0:
            ow = np.pi/nw

            w = np.ones( n )
            for i in range( nw ):
                w[i] = f0 - f1 * np.cos(ow*i)

            for i in range( n-nw,n ):
                w[i] = 1.0 - w[i-n+nw]

            return x * w
        elif nw == 0:
            return x

# Bitwise version

def next_power_of_2(n):
    """
    Return next power of 2 greater than or equal to n
    """
    return 2**(n-1).bit_length()

# PCC2 from Ventosa el al. (2019)

def pcc2(x1, x2, dt, lag0, lagu):
    # Preprocessing

    x1 = x1 - np.mean(x1)
    x2 = x2 - np.mean(x2)
    x1 = taper(x1, 0.05)
    x2 = taper(x2, 0.05)
    N  = len(x1)
    Nz = next_power_of_2( 2*N )

    # Analytic signal and normalization

    xa1 = scipy.signal.hilbert(x1)
    xa2 = scipy.signal.hilbert(x2)
    xa1 = xa1 / np.abs(xa1)
    xa2 = xa2 / np.abs(xa2)

    # Padding zeros

    xa1 = np.append(xa1, np.zeros((Nz-N), dtype=np.complex_))
    xa2 = np.append(xa2, np.zeros((Nz-N), dtype=np.complex_))

    # FFT, correlation and IFFT

    xa1 = np.fft.fft(xa1)
    xa2 = np.fft.fft(xa2)
    amp = xa1 * np.conj(xa2)
    pcc = np.real( np.fft.ifft(amp) ) / N
    pcc = np.fft.ifftshift(pcc)
    tt  = Nz//2 * dt
    t   = np.arange(-tt, tt, dt)

    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]

# PCC2 for autocorrelation

def apcc2(x1, dt, lag0, lagu):
    # Preprocessing
    
    x1 = x1 - np.mean(x1)
    x1 = taper(x1, 0.05)
    N  = len(x1)
    Nz = next_power_of_2( 2*N )

    # Analytic signal and normalization
    start = timer()
    hh = lambda x: signal.hilbert(x1, fftpack.next_fast_len(len(x1)))[:len(x1)]
    xa1 = hh(x1)
    xa1 = xa1 / np.abs(xa1)
    print timer() - start
    # Padding zeros

    xa1 = np.append(xa1, np.zeros((Nz-N), dtype=np.complex_))

    # FFT, correlation and IFFT
    
    xa1 = np.fft.fft(xa1)
    amp = xa1 * np.conj(xa1)
    pcc = np.real( np.fft.ifft(amp) ) / N
    pcc = np.fft.ifftshift(pcc)
    tt  = Nz//2 * dt
    t   = np.arange(-tt, tt, dt)
    
    t = np.around(t, decimals= 4)
    
    return t[(t >= lag0) & (t <= lagu)], pcc[(t >= lag0) & (t <= lagu)]