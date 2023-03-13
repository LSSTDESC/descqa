#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import matplotlib.pyplot as plt
import os
from numba import cuda
import numpy
import math
#from mpi4py import MPI
#comm = MPI.COMM_WORLD
#nranks = comm.Get_size()
#rank = comm.Get_rank()


def get_quantity_labels(bands):
    quantities = ['ra','dec']
    for band in bands:
        quantities.append('mag_'+band)
    return quantities 

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation


def run_test(data, outdir):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    # save plots to the output directory as follows
    plt.figure()
    plt.hist2d(data['ra'],data['dec'],bins=100)
    plt.savefig(os.path.join(outdir,'test_plot.png'))
    plt.close()

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0 
    passed = True

    len_gpus = len(cuda.gpus)
    #proc = MPI.Get_processor_name()
    #allprocs = comm.allgather(proc)
    #count_proc = allprocs.count(proc) # count number of ranks using this process
    #my_proc = allprocs[:rank].count(proc) # which GPU you should use
    #assert(count_proc==len_gpus)
    #cuda.select_device(my_proc)

    # Host code
    data = numpy.ones(256)
    threadsperblock = 256
    blockspergrid = math.ceil(data.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](data)
    print(data)


    return test_result, passed 



