import numpy as np
import time
from scipy import signal
from multiprocessing import Pool, RawArray
from timeit import default_timer as timer

# A global dictionary storing the variables passed from the initializer.
#var_dict = {}

def init_worker(X,Y, X_shape):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['X'] = X
    var_dict['Y'] = Y
    var_dict['X_shape'] = X_shape

def worker_func(i):
    # Simply computes the sum of the i-th row of the input matrix X
    X_np = np.frombuffer(var_dict['X']).reshape(var_dict['X_shape'])
    Y_np = np.frombuffer(var_dict['Y']).reshape(var_dict['X_shape'])
    #time.sleep(1) # Some heavy computations
    acf = signal.correlate(X_np[i,:],Y_np[i,:],mode="full", method="fft")
    print acf.shape
    return acf

# We need this check for Windows to prevent infinitely spawning new child
# processes.
if __name__ == '__main__':
    X_shape = (5, 432000)
    # Randomly generate some data
    data1 = np.random.randn(*X_shape)
    data2 = np.random.randn(*X_shape)
    #print data1
    X = RawArray('d', X_shape[0] * X_shape[1])
    Y = RawArray('d', X_shape[0] * X_shape[1])
    # Wrap X as an numpy array so we can easily manipulates its data.
    X_np = np.frombuffer(X).reshape(X_shape)
    Y_np = np.frombuffer(Y).reshape(X_shape)
    #print X
    # Copy data to our shared array.
    np.copyto(X_np, data1)
    np.copyto(Y_np, data2)
    print X_np
    # Start the process pool and do the computation.
    # Here we pass X and X_shape to the initializer of each worker.
    # (Because X_shape is not a shared variable, it will be copied to each
    # child process.)
    start = timer()
    pool = Pool(processes=10, initializer=init_worker, initargs=(X,Y, X_shape))

    result = pool.map(worker_func, range(X_shape[0]))
    pool.close()
    pool.join()
    #print('Results (pool):\n', np.array(result))
    print np.array(result).shape
    print result
    print timer() - start
    # Should print the same results.
    #print('Results (numpy):\n', np.sum(X_np, 1))