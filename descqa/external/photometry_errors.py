#!/bin/bash/python

'''
This script has the simple functionality needed to tell the example run what data it needs, and then to use that to make a test plot
'''

import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0,'/global/cfs/projectdirs/lsst/groups/SRV/srv_packages/site-packages/')
import qp
import numpy as np
from scipy.stats import chi2

def get_chi2_plots_from_dist(loc,scale,sample,outdir,dist_name):

    ens_n = qp.Ensemble(qp.stats.norm, data=dict(loc=loc, scale=scale))
    cdf_vals = ens_n.cdf(sample)
    
    # create histogram 
    plt.figure()
    a,b,c = plt.hist(cdf_vals.flatten(),bins=30)
    bin_widths = b[1:]-b[:-1]
    
    ngals = len(loc)
    ci = (a - ngals*bin_widths)**2/(ngals*bin_widths)
    x2 = np.sum(ci)

    # create expected value 
    plt.plot(b,ngals*bin_widths[0]*np.ones_like(b),'r--',linewidth=2, label='uniform distribution, chi2/dof = '+str(x2/30))
    plt.xlim([0,1.0])
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(outdir,'PIT_'+dist_name+'.png'))
    plt.close()

    plt.figure()
    plt.hist2d(np.log10(sample),(loc-sample)/scale, bins = [np.linspace(3,4,100),np.linspace(-30,30,100)])
    plt.colorbar()
    plt.plot(np.linspace(3,4,100),np.zeros(100),'w--',linewidth=2)
    plt.ylabel('residual/1sigma error')
    plt.xlabel('log10(flux)-truth')
    plt.savefig(os.path.join(outdir,'fluxerr_'+dist_name+'.png'))

    plt.close()
    # compute chi2 value 
    ci = (a - ngals*bin_widths)**2/(ngals*bin_widths)
    x2 = np.sum(ci)

    return x2/30.

def get_quantity_labels(bands, models,truth):
    quantities=[]
    for band in bands:
        for model in models:
            quantities.append(model+'Flux_flag_'+band)
            quantities.append(model+'Flux_'+band)
            quantities.append(model+'FluxErr_'+band)
        quantities.append('flux_'+band+'_'+truth)
    return quantities 



def run_test(data, outdir, bands, models):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0 
    if 'dp02' in outdir:
        truth = "MATCH"
    else:
        truth = "truth"

    for model in models:
        mask_flags = np.ones(len(data['flux_'+bands[0]+'_'+truth])).astype('bool')        
        for band in bands:
            mask_flags = mask_flags&(~data[model+'Flux_flag_'+band])

        for band in bands:
            loc = data[model+'Flux_'+band][mask_flags]; scale = data[model+'FluxErr_'+band][mask_flags]; sample = data['flux_'+band+'_'+truth][mask_flags]
            test_result = get_chi2_plots_from_dist(loc,scale,sample,outdir,model+'_'+band)

    passed = True#test_result<10 # randomly chosen value for max chi2/dof for now

    return test_result, passed 



def run_test_double(data,data2, outdir, bands, models,truths):
    ''' data is a dictionary containing all the quantities above, 
        outdir is the location for the plot outputs'''

    # return some sort of score and whether the test passed (default to 0 and True)
    test_result = 0

    for model in models:
        mask_flags = np.ones(len(data['flux_'+bands[0]+'_'+truths[0]])).astype('bool')
        for band in bands:
            mask_flags = mask_flags&(~data[model+'Flux_flag_'+band])

        for band in bands:
            loc = data[model+'Flux_'+band][mask_flags]; scale = data[model+'FluxErr_'+band][mask_flags]; sample = data['flux_'+band+'_'+truths[0]][mask_flags]
            test_result = get_chi2_plots_from_dist(loc,scale,sample,outdir,model+'_'+band)

    for model in models:
        mask_flags = np.ones(len(data2['flux_'+bands[0]+'_'+truths[1]])).astype('bool')
        for band in bands:
            mask_flags = mask_flags&(~data2[model+'Flux_flag_'+band])

        for band in bands:
            loc = data2[model+'Flux_'+band][mask_flags]; scale = data2[model+'FluxErr_'+band][mask_flags]; sample = data2['flux_'+band+'_'+truths[1]][mask_flags]
            test_result = get_chi2_plots_from_dist(loc,scale,sample,outdir,'catalog2_'+model+'_'+band)

    passed = True#test_result<10 # randomly chosen value for max chi2/dof for now

    return test_result, passed

