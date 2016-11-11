#!/usr/bin/env python 
import os
import subprocess
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from GalaxyCatalogInterface import loadCatalog
from descqaGlobalConfig import *
from descqaTestConfig import *


# Cosmological parameters
#om = 0.2648
#ob = 0.0448
#h  = 0.71
#s8 = 0.8
#ns = 0.963
#Delta = 700.0

gc=loadCatalog(sim_catalog)
cosmology=gc.get_cosmology()
om = cosmology.Om(gc.redshift)
ob = 0.046 # assume ob is included in om
h  = cosmology.H(gc.redshift).value/100
s8 = 0.816# from paper
ns = 0.96 # from paper
#Delta = 700.0 # default from original halo_mf.py
delta_c = 1.686
fitting_f = 'ST'
#print cosmology.H0,cosmology.Om0,cosmology.Ode0

DESCQAPATH = os.getcwd()
#EXEPATH    = os.path.join(DESCQAPATH, 'amf')
EXEPATH='/project/projectdirs/lsst/descqa/src/catalog/scripts/amf'

# Example call to amf
os.chdir(EXEPATH)
if os.path.exists('analytic.dat'):
    os.remove('analytic.dat')
FNULL = open(os.devnull, 'w')
args=["./amf.exe", "-omega_0", str(om), "-omega_bar", str(ob), "-h", str(h), "-sigma_8", str(s8), \
                    "-n_s", str(ns), "-tf", "EH", "-delta_c", str(delta_c), "-M_min", str(1.0e7), "-M_max", str(1.0e15), \
                    "-z", str(0.0), "-f", fitting_f]
print "Running amf: ", " ".join(args)
#p = subprocess.call(args, stdout=FNULL, stderr=FNULL)
p = subprocess.call(["./amf.exe", "-omega_0", str(om), "-omega_bar", str(ob), "-h", str(h), "-sigma_8", str(s8),
                    "-n_s", str(ns), "-tf", "EH", "-delta_c", str(delta_c), "-M_min", str(1.0e7), "-M_max", str(1.0e15),
                    "-z", str(0.0), "-f", fitting_f], stdout=FNULL, stderr=FNULL)
MassFunc = np.loadtxt('analytic.dat').T
os.chdir(DESCQAPATH)

# MF from a simulation
Nbins = 20
mass = gc.get_quantities("mass", {'zlo': zlo, 'zhi': zhi})
mhist, mbins = np.histogram(np.log10(mass), Nbins)
dM = mbins[1] - mbins[0]
binctr = (mbins[1:] + mbins[:Nbins])/2.0
binwid = mbins[1:] - mbins[:Nbins]
mf     = mhist / (gc.box_size**3 * binwid)
err1   = np.sqrt(mhist) / (gc.box_size**3 * binwid)
err2   = np.sqrt(mhist) / (gc.box_size**3 * binwid)

# Plot
fig = plt.figure(figsize=(4, 4))
ax1 = fig.add_axes((0.17, 0.15, 0.79, 0.80))

ax1.loglog(MassFunc[2]/h, MassFunc[3]*h*h*h, ls="-", label="Analytic", color="#377EB8", alpha=0.8)
ax1.errorbar(10**binctr, mf, yerr=[err1, err2], ls="none",
             label="Sim.", marker="o", ms=5, color="#000000", alpha=0.8)

ax1.legend(loc="best")
ax1.set_ylabel(r"dn/dlog(M) [Mpc$^{-3}$]")
ax1.set_xlabel(r"Mass [Msun]")

plt.grid()
#plt.show()
plt.savefig("mf-test.png", dpi=600)
plt.close()
