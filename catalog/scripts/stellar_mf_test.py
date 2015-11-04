#!/usr/bin/env python

# DESCQA catalog test: read mock galaxy catalog from an N-body simulation and
# generate the stellar mass function.

import numpy as np
import matplotlib.pyplot as mp
import GalaxyCatalog
import sys
from descqaGlobalConfig import *
from descqaTestConfig import *

def usage():
    print("usage: stellar_mf_test.py sim-catalog obs-catalog zlo zhi")
    sys.exit(1)

def gen_stellar_mf(fn, zlo, zhi):
    Nbins = 25      # might need to choose this another way
    gc = GalaxyCatalog.GalaxyCatalog(fn)
    if gc:
        masses = gc.get_quantities("stellar mass function", kw_zlo=zlo, kw_zhi=zhi)
        logm = np.log10(masses)
        mhist, mbins = np.histogram(logm, Nbins)
        binctr = (mbins[1:] + mbins[:Nbins])/2.
        binwid = mbins[1:] - mbins[:Nbins]
        return binctr, binwid, mhist
    else:
        print("could not read galaxy catalog from %s" % fn)
        sys.exit(1)

def read_obs_mf(fn):
    binctr, mhist, mhmax, mhmin = np.genfromtxt(fn, unpack=True, usecols=[0,1,2,3])
    return binctr, mhist, mhmin, mhmax

def write_file(fn, binctr, hist, hmin, hmax, comment=""):
    f = open(fn, 'w')
    if comment:
        f.write('# '+comment+'\n')
    for b, h, hn, hx in zip(binctr, hist, hmin, hmax):
        f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
    f.close()

#------------------------------------------------------------------------------

#if len(sys.argv) != 5:
#    usage()

#sim_catalog = sys.argv[1]
#obs_catalog = sys.argv[2]
#zlo         = sys.argv[3]
#zhi         = sys.argv[4]

sbinctr, sbinwid, shist = gen_stellar_mf(sim_catalog, zlo, zhi)
obinctr, ohist, ohmin, ohmax = read_obs_mf(obs_catalog)

shmin = shist - np.sqrt(shist)
shmax = shist + np.sqrt(shist)

mp.step(sbinctr, shist, where="mid", label=sim_catalog)
mp.errorbar(obinctr, ohist, yerr=[ohmin, ohmax], label=obs_catalog, fmt='o')
mp.legend()
mp.title(r'Stellar mass function, $%4.2f < z < %4.2f$' % (zlo, zhi))
mp.xlabel(r'$\log M_*\ (M_\odot)$')
mp.ylabel(r'$dN/d\log M\ (M_\odot^{-1})$')
mp.savefig("stellar_mf_test.png")

write_file("theory_output.txt", sbinctr, shist, shmin, shmax)
write_file("plot_output.txt", obinctr, ohist, ohmin, ohmax)
