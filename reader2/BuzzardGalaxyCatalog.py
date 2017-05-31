"""
Buzzard galaxy catalog class.
"""
from __future__ import division
import os
import numpy as np
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from .BaseGalaxyCatalog import BaseGalaxyCatalog

__all__ = ['BuzzardGalaxyCatalog']


class BuzzardGalaxyCatalog(BaseGalaxyCatalog):
    """
    Argonne galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGalaxyCatalog class.
    """

    def _subclass_init(self, catalog_dir, base_catalog_dir=os.curdir, **kwargs):

        self._pre_filter_quantities = {'original_healpixel'}

        self._quantity_modifiers = {
            'redshift': 'Z',
        }

        self._catalog_dir = os.path.join(base_catalog_dir, catalog_dir)
        self.cosmology = None
        self._native_quantities = set(next(self._iter_native_dataset())[1].names)
        self._native_quantities.add('original_healpixel')


    def _iter_native_dataset(self, pre_filters=None):
        for i in xrange(768):
            if pre_filters and not all(f[0](*([i]*(len(f)-1))) for f in pre_filters):
                continue

            fname = os.path.join(self._catalog_dir, 'Chinchilla-0_lensed.{}.fits'.format(i))
            if not os.path.isfile(fname):
                continue
            f = fits.open(fname)
            yield i, f[1].data
            f.close()


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        healpix, fits_data = dataset
        if native_quantity == 'original_healpixel':
            data = np.empty(fits_data.shape, np.int)
            data.fill(healpix)
            return data
        return fits_data[native_quantity]

