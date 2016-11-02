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
        print("Could not read galaxy catalog from " + sim_catalog)
        return

    flag = (gc.get_quantities("stellar_mass", {}) >= sm_cut)
    x = gc.get_quantities("positionX", {})
    flag &= np.isfinite(x)
    
    x = x[flag]
    y = gc.get_quantities("positionY", {})[flag]
    z = gc.get_quantities("positionZ", {})[flag]
    vz = gc.get_quantities("velocityZ", {})[flag]
    
    try:
        h = gc.cosmology.H0.value/100.0
    except AttributeError:
        print("Make sure `cosmology` and `redshift` properties are set. Using default value h=0.702...")
        h = 0.702

    vz /= (100.0*h)
    z += vz
    
    return np.remainder(np.vstack((x,y,z)).T, gc.box_size), gc.box_size, h
    

def load_obs_catalog(obs_catalog):
    pass

def make_plot(ax, rp, wp, wp_err, label, save_output=None):
    if save_output is not None:
        np.savetxt(save_output, np.vstack((rp, wp, wp_err)).T, header='rp wp wp_err')
    l = ax.loglog(rp, wp, label=label, lw=1.5)[0]
    ax.fill_between(rp, wp+wp_err, np.where(wp > wp_err, wp - wp_err, 0.01), alpha=0.2, color=l.get_color(), lw=0)


#---------------------------------------------------------------------------
points, box_size, h = load_sim_catalog(sim_catalog, sm_cut)
wp, wp_cov = projected_correlation(points, rbins, zmax, box_size, njack)
rp = np.sqrt(rbins[1:]*rbins[:-1])
wp_err = np.sqrt(np.diag(wp_cov))

fig, ax = plt.subplots()

make_plot(ax, rp, wp, wp_err, sim_name, 'plot_output.txt')

# load mb2 wp(rp), use this to validate
rp, wp, wp_err = np.loadtxt(obs_catalog).T
make_plot(ax, rp, wp, wp_err, obs_name, 'validation_output.txt')

# load sdss wp(rp), just for comparison
rp, wp, wp_err = np.loadtxt(sdss_wprp).T
make_plot(ax, rp, wp, wp_err, 'SDSS')

ax.set_xlim(0.1, 50.0)
ax.set_ylim(1.0, 3.0e3)
ax.set_xlabel(r'$r_p \; {\rm [Mpc]}$')
ax.set_ylabel(r'$w_p(r_p) \; {\rm [Mpc]}$')
ax.set_title(r'Projected correlation function ($M_* > {0:.2E} \, {{\rm M}}_\odot$)'.format(sm_cut))
ax.legend(loc='best', frameon=False)
plt.savefig('wprp.png')

