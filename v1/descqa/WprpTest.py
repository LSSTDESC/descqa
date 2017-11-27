from __future__ import division, print_function
import os
import numpy as np
from warnings import warn

from .ValidationTest import *
from helpers.CorrelationFunction import projected_correlation


class WprpTest(ValidationTest):
    """
    validation test class object to compute project 2-point correlation function wp(rp)
    """
    _plot_config = dict(\
        xlabel=r'$r_p \; [{\rm Mpc}]$',
        ylabel=r'$w_p(r_p) \; [{\rm Mpc}]$',
        xlim=(0.1, 30.0),
        ylim=(0.1, 2000.0),
    )
    _required_quantities = {'stellar_mass', 'positionX', 'positionY', 'positionZ', 'velocityZ'}
    _available_observations = {'SDSS', 'MBII'}
    _default_kwargs = {
        'observation': 'SDSS',
        'zlo': 0.0,
        'zhi': 1000.0,
        'jackknife_nside': 10,
        'zmax': 40.0,
        'sm_cut': 10.0**9.8,
    }

    def _subclass_init(self, **kwargs):

        #set validation data information

        self._import_kwargs(kwargs, 'datafile', required=True)
        self._import_kwargs(kwargs, 'sm_cut', func=float, required=True)
        self._import_kwargs(kwargs, 'zmax', func=float, required=True)

        raw_data = np.loadtxt(os.path.join(self._base_data_dir, self._datafile))
        rp = raw_data[:,0]
        wp = raw_data[:,1]
        wp_cov = raw_data[:,2:]
        wp_err = np.sqrt(np.diag(wp_cov))
        self._validation_data = {'x': rp, 'y':wp, 'y+':wp+wp_err, 'y-':wp-wp_err, 'cov':wp_cov}


    def _calc_catalog_result(self, gc):
        try:
            h = gc.cosmology.H0.value/100.0
        except AttributeError:
            h = 0.702
            msg = 'Make sure `cosmology` and `redshift` properties are set. Using default value h=0.702...'
            warn(msg)

        # convert arguments
        sm_cut = self._sm_cut/(h*h)
        rbins = self._bins/h
        zmax = self._zmax/h
        njack = self._jackknife_nside

        # load catalog
        flag = (gc.get_quantities("stellar_mass", self._zfilter) >= sm_cut)
        x = gc.get_quantities("positionX", self._zfilter)
        flag &= np.isfinite(x)

        x = x[flag]
        y = gc.get_quantities("positionY", self._zfilter)[flag]
        z = gc.get_quantities("positionZ", self._zfilter)[flag]
        vz = gc.get_quantities("velocityZ", self._zfilter)[flag]

        vz /= (100.0*h)
        z += vz
        del vz

        # calc wp(rp)
        points = np.remainder(np.vstack((x, y, z)).T, gc.box_size)
        wp, wp_cov = projected_correlation(points, rbins, zmax, gc.box_size, njack)
        rp = np.sqrt(rbins[1:]*rbins[:-1])
        wp_err = np.sqrt(np.diag(wp_cov))
        return {'x': rp, 'y':wp, 'y+':wp+wp_err, 'y-':wp-wp_err, 'cov':wp_cov}

