from __future__ import division, print_function

import os
import subprocess

import numpy as np

from ValidationTest import *


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


class StellarMassHaloMassTest(ValidationTest):
    """
    validation test class object to compute stellar mass halo mass relation
    """
    _plot_config = dict(\
        xlabel=r'$M_{\rm halo} \;  [{\rm M}_\odot]$',
        ylabel=r'$\langle M_* \rangle \; [{\rm M}_\odot]$',
        xlim=(1.0e8, 1.0e15),
        ylim=(1.0e7, 1.0e13),
        ylim_lower=(0.1, 10.0),
    )

    _required_quantities = {'mass', 'stellar_mass', 'parent_halo_id', 'positionX', 'positionY', 'positionZ'}

    _available_observations = {'MassiveBlackII'}

    _default_kwargs = {
        'zlo': 0,
        'zhi': 1000.0,
        'jackknife_nside': 5,
    }

    def _subclass_init(self, **kwargs):
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
        fn = os.path.join(self._base_data_dir, 'MASSIVEBLACKII/StellarMass_HaloMass/tab_new.txt')
        self._validation_data = dict(zip(('x', 'y', 'y-', 'y+'), np.loadtxt(fn, unpack=True, usecols=(0,1,3,4))))
        self._validation_data['cov'] = np.diag(((self._validation_data['y+']-self._validation_data['y-'])*0.5)**2.0)
        self._validation_name = 'MBII (validation)'


    def _get_quantities_from_catalog(self, galaxy_catalog):
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


    def _calc_catalog_result(self, galaxy_catalog):
        """
        calculate the stellar mass - halo mass relation in bins

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get quantities from galaxy catalog
        quantities = self._get_quantities_from_catalog(galaxy_catalog)

        #sort halo mass
        s = quantities['hm'].argsort()
        for k in quantities:
            quantities[k] = quantities[k][s]
        del s

        res = {'x': np.sqrt(self._bins[1:]*self._bins[:-1])}

        #get errors from jackknife samples if requested
        if self._jackknife_nside > 0:
            masses = np.vstack((quantities['hm'], quantities['sm'])).T
            jack_indices = CalcStats.get_subvolume_indices(quantities['x'], quantities['y'], quantities['z'], \
                    galaxy_catalog.box_size, self._jackknife_nside)
            njack = self._jackknife_nside**3
            res['y'], _, covariance = CalcStats.jackknife(masses, jack_indices, njack, \
                    lambda arr: mean_y_in_x_bins(arr[:,1], arr[:,0], self._bins))
            yerr = np.sqrt(np.diag(covariance))
            res.update({'y-':res['y']-yerr, 'y+':res['y']+yerr, 'cov':covariance})
        else:
            res['y'] = mean_y_in_x_bins(quantities['sm'], quantities['hm'], self._bins)

        return res


