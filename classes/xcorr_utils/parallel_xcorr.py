import numpy as np
import time
from scipy import signal
from multiprocessing import Pool, RawArray
from timeit import default_timer as timer

# A global dictionary storing the variables passed from the initializer.
#
var_dict = {}

def init_worker(X,Y, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    #var_dict = {}
    var_dict['X'] = X
    var_dict['Y'] = Y
    var_dict['X_shape'] = X_shape

def worker_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    Y_np = np.frombuffer(var_dict['Y']).reshape(var_dict['X_shape'])
    #time.sleep(1) # Some heavy computations
    acf = signal.correlate(X_np[i,:],Y_np[i,:],mode="full", method="fft")
    tcorr = np.arange(-var_dict['X_shape'][1] + 1, var_dict['X_shape'][1])
    dN = np.where(np.abs(tcorr) <= 600*5)[0]
    return acf[dN]