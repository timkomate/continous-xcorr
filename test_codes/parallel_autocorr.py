import numpy as np
import itertools
from scipy import signal
from multiprocessing import Pool #  Process pool
from multiprocessing import sharedctypes
import matplotlib.pyplot as plt
from timeit import default_timer as timer

col = 432000
row = 365
block_size = 4

X = np.random.random((row, col))
Y = np.random.random((row, col))

result = np.ctypeslib.as_ctypes(np.zeros((row, 2*col-1)))
shared_array = sharedctypes.RawArray(result._type_, result)

res2 = np.zeros((row, 2*col - 1))

def autocorr(row):
    print row
    tmp = np.ctypeslib.as_array(shared_array)
    print "tmp:", tmp.nbytes
    tmp[row,:] = signal.correlate(X[row,:], Y[row,:], mode="full", method="fft")


window_idxs = np.arange(0,X.shape[0], 1)

start = timer()
p = Pool(4)
p.map(autocorr, window_idxs)
result = np.ctypeslib.as_array(shared_array)
#p.close()
#p.join()
#print res
print timer() - start

"""start = timer()
for row in range(X.shape[0]):
    #print row
    res2[row,:] = signal.correlate(X[row,:], X[row,:], mode="full", method = "fft")
print timer() - start
print(np.array_equal(res2, result))"""

