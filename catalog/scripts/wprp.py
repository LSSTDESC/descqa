#!/usr/bin/env python

import numpy as np

import matplotlib as mpl
mpl.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

from GalaxyCatalogInterface import loadCatalog
from descqaGlobalConfig import *
from descqaTestConfig import *

from helpers.CorrelationFunction import projected_correlation

def load_sim_catalog(sim_catalog, sm_cut):
    gc = loadCatalog(sim_catalog)
    if not gc:
        print("could not read galaxy catalog from %s" % fn)
        return
    flag = (gc.get_quantities("stellar_mass", {}) >= sm_cut)
    x = gc.get_quantities("positionX", {})[flag]
    y = gc.get_quantities("positionY", {})[flag]
    z = gc.get_quantities("positionZ", {})[flag]
    vz = gc.get_quantities("velocityZ", {})[flag]
    vz /= 100.0
    z += vz
    return np.remainder(np.vstack((x,y,z)).T, gc.box_size), gc.box_size
    

def load_obs_catalog(obs_catalog):
    pass

#---------------------------------------------------------------------------
points, box_size = load_sim_catalog(sim_catalog, sm_cut)
print(box_size)
if (not np.isfinite(points).all()):
    raise ValueError('something is very wrong: nan points')
if (points >= box_size).any() or (points < 0.0).any():
    raise ValueError('something is very wrong: points outside box')
wp, wp_cov = projected_correlation(points, rbins, zmax, box_size, njack)
rp = np.sqrt(rbins[1:]*rbins[:-1])
wp_err = np.sqrt(np.diag(wp_cov))

np.savetxt('plot_output.txt', np.vstack((rp, wp, wp_err)).T, header='rp wp wp_err')
l = plt.loglog(rp, wp, label=sim_name);
plt.fill_between(rp, wp+wp_err, np.where(wp > wp_err, wp-wp_err, 1.0), alpha=0.3, color=l[0].get_color(), lw=0);

rp, wp, wp_err = np.loadtxt(obs_catalog, unpack=True)
np.savetxt('theory_output.txt', np.vstack((rp, wp, wp_err)).T, header='rp wp wp_err')
l = plt.loglog(rp, wp, label=obs_name);
plt.fill_between(rp, wp+wp_err, np.where(wp > wp_err, wp-wp_err, 1.0), alpha=0.3, color=l[0].get_color(), lw=0);

plt.ylim(1.0, None)
plt.xlabel(r'$r_p$ [Mpc/$h$]')
plt.ylabel(r'$w_p(r_p)$ [Mpc/$h$]')
plt.title('Projected correlation function (M* > {0:E})'.format(sm_cut))
plt.legend(loc='best', frameon=False)
plt.savefig('wprp.png')

