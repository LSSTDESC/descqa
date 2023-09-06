#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import matplotlib.pyplot as plt
import os
import numpy
import math


import sys
sys.path.insert(0,'/global/cfs/projectdirs/lsst/groups/SRV/srv_packages/site-packages/')

import pandas as pd
import pzflow
from pzflow import Flow, FlowEnsemble
from pzflow.distributions import Uniform
from pzflow.bijectors import Chain, StandardScaler, NeuralSplineCoupling, ColorTransform, InvSoftplus, RollingSplineCoupling, ShiftBounds

def get_training_set(data, fract=0.01):
    data_small={}
    test_data = data[list(data.keys())[0]]
    nl = int(len(test_data)*fract)
    for key in data.keys():
        data_small[key] = data[key][:nl]
    return data_small
        

def get_quantity_labels(bands):
    quantities = []
    for band in bands:
        quantities.append(band)
    quantities.append('redshift_true')
    return quantities

def run_test(data, outdir):
    ''' data is a dictionary containing all the quantities above,
        outdir is the location for the plot outputs'''

    # save plots to the output directory as follows
    # plt.figure()
    # plt.hist2d(data['mag_g_lsst'],data['mag_r_lsst'],bins=100)
    # plt.savefig(os.path.join(outdir,'TLM_test_plot.png'))
    # plt.close()


    training_data = get_training_set(data, fract=0.01)

    df = pd.DataFrame(training_data)
    data_columns = ["redshift_true"]
    conditional_columns = df.columns.drop('redshift_true')
    ndcol = len(data_columns)
    ncond = len(conditional_columns)
    ndim = ndcol + ncond

    bijector = Chain(
    # InvSoftplus(z_col, sharpness),
    ShiftBounds(0, 6, B=4),
    RollingSplineCoupling(nlayers=1, n_conditions=ncond, B=6),
  )
    latent = Uniform(input_dim=ndcol, B=7)
    ens = FlowEnsemble(data_columns = data_columns,
                           conditional_columns = conditional_columns,
                           bijector = bijector,
                           latent = latent)
    # df should be replaced with a random subset
    df_subset = df

    loss = ens.train(df_subset, convolve_errs=False, epochs=100, verbose=True)
    ens.save('TheLastMetric_test_srv.pkl')
    just_tav = ens.log_prob(df_subset[conditional_columns])

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = np.mean(just_tav)
    passed = True

    return test_result, passed

