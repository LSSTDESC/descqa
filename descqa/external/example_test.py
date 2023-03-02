#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import matplotlib.pyplot as plt
import os

def get_quantity_labels(bands):
    quantities = ['ra','dec']
    for band in bands:
        quantities.append('mag_'+band)
    return quantities 

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

    return test_result, passed 



