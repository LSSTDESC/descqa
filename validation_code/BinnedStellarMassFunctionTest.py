from __future__ import division, print_function
import os
import numpy as np

from ValidationTest import *


class BinnedStellarMassFunctionTest(ValidationTest):
    """
    validation test class object to compute stellar mass function bins
    """

    _plot_config = dict(\
        xlabel=r'$M^* \; [{\rm M}_\odot]$',
        ylabel=r'$dn\,/\,(dV\,d\log M) \; [{\rm Mpc}^{-3}\,{\rm dex}^{-1}]$',
        xlim=(1.0e7, 1.0e13),
        ylim=(1.0e-7, 10.0),
        ylim_lower=(0.1, 10.0),
    )
    _required_quantities = {'stellar_mass', 'positionX', 'positionY', 'positionZ'}
    _available_observations = {'LiWhite2009', 'MassiveBlackII'}
    _default_kwargs = {
        'observation': 'LiWhite2009',
        'zlo': 0.0,
        'zhi': 1000.0,
        'jackknife_nside': 5,
    }

    def _subclass_init(self, **kwargs):
        """
        load tabulated stellar mass function data
        """
        columns = {'LiWhite2009': (0,5,6), 'MassiveBlackII': (0,1,2)}.get(self._observation)
        fn = os.path.join(self._base_data_dir, 'LIWHITE/StellarMassFunction/massfunc_dataerr.txt')

        #column 1: stellar mass bin center
        #column 2: number density
        #column 3: 1-sigma error
        binctr, mhist, merr = np.loadtxt(fn, unpack=True, usecols=columns)
        self._validation_data = {'x':binctr, 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':np.diag(merr*merr)}


    def _get_quantities_from_catalog(self, galaxy_catalog):
        """
        obtain the masses and mask fom the galaxy catalog

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        #get stellar masses from galaxy catalog
        sm = galaxy_catalog.get_quantities("stellar_mass", self._zfilter)
        x = galaxy_catalog.get_quantities("positionX", self._zfilter)
        y = galaxy_catalog.get_quantities("positionY", self._zfilter)
        z = galaxy_catalog.get_quantities("positionZ", self._zfilter)

        #remove non-finite or negative numbers
        mask = np.isfinite(sm)
        mask &= (sm > 0)
        mask &= np.isfinite(x)
        mask &= np.isfinite(y)
        mask &= np.isfinite(z)

        return dict(mass=sm[mask], x=x[mask], y=y[mask], z=z[mask])


    def _calc_catalog_result(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get stellar masses from galaxy catalog
        quantities = self._get_quantities_from_catalog(galaxy_catalog)

        #histogram points and compute bin positions
        mhist = np.histogram(quantities['mass'], bins=self._bins)[0].astype(float)
        summass = np.histogram(quantities['mass'], bins=self._bins, weights=quantities['mass'])[0]
        binwid = np.log10(self._bins[1:]/self._bins[:-1])
        binctr = np.sqrt(self._bins[1:]*self._bins[:-1])
        has_mass = (mhist > 0)
        binctr[has_mass] = (summass/mhist)[has_mass]

        #count galaxies in log bins
        #get errors from jackknife samples if requested
        if self._jackknife_nside > 0:
            jack_indices = CalcStats.get_subvolume_indices(quantities['x'], quantities['y'], quantities['z'], \
                    galaxy_catalog.box_size, self._jackknife_nside)
            njack = self._jackknife_nside**3
            mhist, _, covariance = CalcStats.jackknife(quantities['mass'], jack_indices, njack, \
                    lambda m, scale: np.histogram(m, bins=self._bins)[0]*scale, \
                    full_args=(1.0,), jack_args=(njack/(njack-1.0),))
        else:
            covariance = np.diag(mhist)

        #calculate number differential density
        vol = galaxy_catalog.box_size**3.0
        mhist /= (binwid * vol)
        covariance /= (vol*vol)
        covariance /= np.outer(binwid, binwid)
        merr = np.sqrt(np.diag(covariance))

        return {'x':binctr, 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':covariance}

