from __future__ import (division, print_function, absolute_import)
import os
import subprocess
from warnings import warn
import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ValidationTest import ValidationTest, TestResult
import CalcStats
from BinnedStellarMassFunctionTest import *

__all__ = ['StellarMassHaloMassTest', 'plot_summary']


class StellarMassHaloMassTest(BinnedStellarMassFunctionTest):
    """
    validation test class object to compute stellar mass halo mass relation
    """
    output_config = dict(\
            catalog_output_file='catalog_smhm.txt',
            validation_output_file='validation_smhm.txt',
            covariance_matrix_file='covariance_smhm.txt',
            log_file='log_smhm.txt',
            plot_file='plot_smhm.png',
            plot_title='Stellar mass - halo mass relation',
            xaxis_label=r'$\log \, M_{\rm halo} / M_\odot$',
            yaxis_label=r'$\langle M^{*} \rangle / M_\odot$',
            plot_validation_as_line=False,
    )

    required_quantities = ('mass', 'stellar_mass', 'parent_halo_id', 'positionX', 'positionY', 'positionZ')

    available_observations = ('MassiveBlackII',)

    default_kwargs = {
            'zlo': 0,
            'zhi': 1000.0,
            'summary_statistic': 'chisq',
            'jackknife_nside': 5,
    }

    enable_interp_validation = False


    def load_validation_data(self):
        """
        load tabulated stellar mass halo mass function data
        """

        #column 1: halo mass bin center
        #column 2: mean stellar mass
        #column 4: mean stellar mass - error (on mean)
        #column 5: mean stellar mass + error (on mean)
        #column 6: bin minimum
        #column 7: bin maximum
        #column 8: 1-sigma error
        #column 9: 16th percentile
        #column 11: 84th percentile
        fn = os.path.join(self.base_data_dir, 'MASSIVEBLACKII/StellarMass_HaloMass/tab_new.txt')
        self.validation_data = dict(zip(('x', 'y', 'y-', 'y+'), np.loadtxt(fn, unpack=True, usecols=(0,1,3,4))))
        self.validation_data['x'] = np.log10(self.validation_data['x'])
        return self.validation_data


    def get_quantities_from_catalog(self, galaxy_catalog):
        """
        obtain the masses and mask fom the galaxy catalog

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        #get stellar masses from galaxy catalog
        hm = galaxy_catalog.get_quantities("mass", self._zfilter)
        sm = galaxy_catalog.get_quantities("stellar_mass", self._zfilter)
        x = galaxy_catalog.get_quantities("positionX", self._zfilter)
        y = galaxy_catalog.get_quantities("positionY", self._zfilter)
        z = galaxy_catalog.get_quantities("positionZ", self._zfilter)
        pid = galaxy_catalog.get_quantities("parent_halo_id", self._zfilter)

        #remove non-finite or negative numbers
        mask = np.isfinite(hm)
        mask &= (hm > 0)
        mask = np.isfinite(sm)
        mask &= (sm > 0)
        mask &= np.isfinite(x)
        mask &= np.isfinite(y)
        mask &= np.isfinite(z)
        mask &= (pid == -1)

        return dict(hm=hm[mask], sm=sm[mask], x=x[mask], y=y[mask], z=z[mask])


    def calc_catalog_result(self, galaxy_catalog):
        """
        calculate the stellar mass - halo mass relation in bins

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get quantities from galaxy catalog
        quantities = self.get_quantities_from_catalog(galaxy_catalog)

        #sort halo mass
        s = quantities['hm'].argsort()
        for k in quantities:
            quantities[k] = quantities[k][s]
        del s

        res = {'x':(self.bins[1:] + self.bins[:-1])*0.5}

        #get errors from jackknife samples if requested
        if self.jackknife_nside > 0:
            masses = np.vstack((np.log10(quantities['hm']), quantities['sm'])).T
            jack_indices = CalcStats.get_subvolume_indices(quantities['x'], quantities['y'], quantities['z'], \
                    galaxy_catalog.box_size, self.jackknife_nside)
            njack = self.jackknife_nside**3
            res['y'], _, covariance = CalcStats.jackknife(masses, jack_indices, njack, \
                    lambda arr: mean_y_in_x_bins(arr[:,1], arr[:,0], self.bins))
            yerr = np.sqrt(np.diag(covariance))
            res.update({'y-':res['y']-yerr, 'y+':res['y']+yerr, 'cov':covariance})
        else:
            res['y'] = mean_y_in_x_bins(quantities['sm'], np.log10(quantities['hm']), self.bins)

        return res


plot_summary = StellarMassHaloMassTest.plot_summary


def mean_y_in_x_bins(y, x, bins, sorter=None):
    y = np.asanyarray(y)
    k = np.searchsorted(x, bins, sorter=sorter)
    res = []
    for i, j in zip(k[:-1], k[1:]):
        if j == i:
            res.append(np.nan)
        else:
            s_this = slice(i, j) if sorter is None else sorter[i:j]
            res.append(y[s_this].mean())
    return np.array(res)
