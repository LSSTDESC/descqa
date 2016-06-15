#!/usr/bin/env python

# DESCQA catalog test: read mock galaxy catalog from an N-body simulation and
# generate the average stellar mass - halo mass relation. Plot it and observational.

import numpy as np
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as mp
from astropy import units as u
import sys
from GalaxyCatalogInterface import loadCatalog
from descqaGlobalConfig import *
from descqaTestConfig import *

def usage():
    print("usage: stellarmass_halomass_test.py sim-catalog obs-catalog zlo zhi")
    sys.exit(1)

def gen_stellarmass_halomass(fn, zlo, zhi):
    Nbins = 25      # might need to choose this another way
    gc = loadCatalog(fn)
    if gc:
        stellarmasses = gc.get_quantities("stellar_mass", {'zlo': zlo, 'zhi': zhi})
        masses = (gc.get_quantities("mass", {'zlo': zlo, 'zhi': zhi}))
        nan_masses = np.isnan(masses)
        stellarmasses = stellarmasses[~nan_masses]
        masses = masses[~nan_masses]
        logm = np.log10(masses)
        mhist, mbins = np.histogram(logm, Nbins)
        binctr = (mbins[1:] + mbins[:Nbins])/2.
        binwid = mbins[1:] - mbins[:Nbins]
        avg_stellarmass = np.zeros(Nbins)
        avg_stellarmasserr = np.zeros(Nbins)
        for i in range(Nbins):
            binsmass = stellarmasses[(logm >= mbins[i]) & (logm < mbins[i+1])]
            avg_stellarmass[i] = np.mean(binsmass) 
            avg_stellarmasserr[i] = np.std(binsmass)/np.sqrt(len(binsmass))
        avg_stellarmassmin = avg_stellarmass - avg_stellarmasserr
        avg_stellarmassmax = avg_stellarmass + avg_stellarmasserr
        print 'binctr = ', binctr
        print 'avgstellarmass = ', avg_stellarmass
        return binctr, binwid, avg_stellarmass, avg_stellarmassmin, avg_stellarmassmax
    else:
        print("could not read galaxy catalog from %s" % fn)
        sys.exit(1)

def read_obs_stellarmass_halomass(fn, usecols):
    binctr, avgstellarmass, avgstellarmassmin, avgstellarmassmax = np.loadtxt(fn, unpack=True, usecols=usecols) # if real errorbar available
    return binctr, avgstellarmass, avgstellarmassmin, avgstellarmassmax

def write_file(fn, binctr, avg_smass, avg_smassmin, avg_smassmax, comment=""):
    f = open(fn, 'w')
    if comment:
        f.write('# '+comment+'\n')
    for b, savg, savgmin, savgmax in zip(binctr, avg_smass, avg_smassmin, avg_smassmax):
        f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, savg, savgmin, savgmax))
    f.close()

#------------------------------------------------------------------------------

sbinctr, sbinwid, savg_smass, savg_smassmin, savg_smassmax = gen_stellarmass_halomass(sim_catalog, zlo, zhi)
obinctr, oavg_smass, oavg_smassmin, oavg_smassmax = read_obs_stellarmass_halomass(obs_catalog_avgstrm, obs_catalog_avgstrm_usecols)

sim_catalog_short = os.path.basename(sim_catalog)
sim_catalog_short = sim_catalog_short[:15] + (sim_catalog_short[15:] and '...')
obs_catalog_short = os.path.basename(obs_catalog)
obs_catalog_short = obs_catalog_short[:15] + (obs_catalog_short[15:] and '...')

mp.step(sbinctr, savg_smass, where="mid", label=sim_catalog_short)
mp.errorbar(obinctr, oavg_smass, yerr=[oavg_smassmin-oavg_smass, oavg_smassmax-oavg_smass], label='validation', fmt='o')
mp.yscale('log')
mp.legend(loc='best', frameon=False)
mp.title(r'Average Stellar mass - Halo mass function')
mp.xlabel(r'$\log M_{halo}\ (M_\odot)$')
mp.ylabel(r'Average $M_{stellarmass}\ (M_\odot)$')
mp.savefig("stellarmass_halomass_test.png")

write_file("validation_output_avgstellarmass.txt", sbinctr, savg_smass, savg_smassmin, savg_smassmax)
write_file("catalog_output_avgstellarmass.txt", obinctr, oavg_smass, oavg_smassmin, oavg_smassmax)
