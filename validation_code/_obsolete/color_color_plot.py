#!/usr/bin/env python
from GalaxyCatalogInterface import loadCatalog
from descqaGlobalConfig import *
from descqaTestConfig import *
from matplotlib.path import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def usage():
    print ("usage: color_color_plot.py sim-catalog obs-catalog")
    sys.exit(1)


def gen_colors(fn):
    gc = loadCatalog(fn)
    if gc:
        mags = gc.get_quantities(["SDSS_u:rest:", "SDSS_g:rest:",
                                  "SDSS_r:rest:", "SDSS_i:rest:",
                                  "SDSS_z:rest:"], {})
#        mags = gc.get_quantities(["SDSS_u:observed:", 
#                                  "SDSS_g:observed:",
#                                  "SDSS_r:observed:", 
#                                  "SDSS_i:observed:",
#                                  "SDSS_z:observed:"], {})

        mags = np.array(mags)
        colors = [mags[i]-mags[i+1] for i in xrange(4)]
        colors = np.array(colors)
        return colors
    else:
        print ("could not read galaxy catalog from %s" % fn)
        sys.exit(1)


def read_obs_colors(fn):
    
    obs_mags = np.genfromtxt(fn, unpack=True)
    obs_colors = np.array([obs_mags[i]-obs_mags[i+1] for i in xrange(4)])
    return obs_colors

# ---------------------------------------------------------------------------

sim_colors = gen_colors(sim_catalog)
c_labels = ['u-g', 'g-r', 'r-i', 'i-z']

sim_catalog_short = os.path.basename(sim_catalog)
sim_catalog_short = sim_catalog_short[:15] + (sim_catalog_short[15:] and '...')
obs_catalog_short = os.path.basename(obs_catalog)
obs_catalog_short = obs_catalog_short[:15] + (obs_catalog_short[15:] and '...')

file_name = 'sdss_specgals_mags.dat.gz'

fig = plt.figure(figsize=(16,16))

sdss_colors = read_obs_colors(str(obs_catalog + file_name))
x_limits = [[-1, 5], [-0.5, 2.5], [-1, 1]]
y_limits = [[-2, 4], [-0.5, 1.5], [-1, 1]]
bins = [60, [30,20], 20]

# Plotting methods based upon scatter_countour.py from astroML
# https://github.com/astroML/astroML/blob/master/astroML/
# plotting/scatter_contour.py

for colorNum in range(0, len(c_labels)-1):
    plt.subplot(2,2,colorNum+1)
    H_obs, x_obs_bins, y_obs_bins = np.histogram2d(sdss_colors[colorNum], 
                                                   sdss_colors[colorNum+1], 
                                                   bins=bins[colorNum],
                                                   range = [x_limits[colorNum],
                                                            y_limits[colorNum]])
    H_obs=H_obs/np.max(H_obs)
    xb_obs, yb_obs = np.meshgrid(x_obs_bins[:-1], y_obs_bins[:-1])
    xb_obs+=(0.5*(x_obs_bins[1]-x_obs_bins[0]))
    yb_obs+=(0.5*(y_obs_bins[1]-y_obs_bins[0]))
    levels = [.001, .01, .1, .5, .9]
    extent = [x_obs_bins[0], x_obs_bins[-1], y_obs_bins[0], y_obs_bins[-1]]
    c1 = plt.contour(xb_obs, yb_obs, H_obs.T, colors='k', 
                     lw=20, levels=levels, extent=extent, linewidth=30)

    H_sim, x_sim_bins, y_sim_bins = np.histogram2d(sim_colors[colorNum], 
                                                   sim_colors[colorNum+1], 
                                                   bins = [x_obs_bins, y_obs_bins])
    H_sim=H_sim/np.max(H_sim)
    xb_sim, yb_sim = np.meshgrid(x_sim_bins[:-1], y_sim_bins[:-1])
    xb_sim+=(0.5*(x_sim_bins[1]-x_sim_bins[0]))
    yb_sim+=(0.5*(y_sim_bins[1]-y_sim_bins[0]))
    levels = [.0001, .001, .01, .1, .5, .9]
    extent = [x_sim_bins[0], x_sim_bins[-1], y_sim_bins[0], y_sim_bins[-1]]
    c2 = plt.contour(xb_sim, yb_sim, H_sim.T, colors='r', lw=20, 
                     levels=levels, extent=extent, linewidth=30)
    c1.collections[0].set_label('Truth Catalog')
    c2.collections[0].set_label('Sim Catalog')

    if colorNum == 0:
        plt.legend(loc=2)
    plt.xlabel(c_labels[colorNum])
    plt.ylabel(c_labels[colorNum+1])
    plt.xlim(x_limits[colorNum])
    plt.ylim(y_limits[colorNum])
plt.suptitle('Observed Color-Color Plots')
plt.subplots_adjust(top=0.95)
plt.savefig('color_color.png')
