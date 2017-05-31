"""
Argonne galaxy catalog class.
"""
from __future__ import division
import os
import numpy as np
import h5py
from astropy.cosmology import FlatLambdaCDM
from .BaseGalaxyCatalog import BaseGalaxyCatalog

__all__ = ['GalacticusGalaxyCatalog']

class GalacticusGalaxyCatalog(BaseGalaxyCatalog):
    """
    Argonne galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by BaseGalaxyCatalog class.
    """

    def _subclass_init(self, filename, base_catalog_dir=os.curdir, **kwargs):

        self._pre_filter_quantities = {'cosmological_redshift'}

        self._quantity_modifiers = {
            'stellar_mass': (lambda x: x**10.0, 'log_stellarmass'),
        }

        self._file = os.path.join(base_catalog_dir, filename)

        with h5py.File(self._file, 'r') as fh:
            self.cosmology = FlatLambdaCDM(
                H0=fh['cosmology'].attrs['H_0'],
                Om0=fh['cosmology'].attrs['Omega_Matter'],
            )

            for k in fh:
                if k != 'cosmology':
                    self._native_quantities = set(fh[k].keys())
                    break

        self._native_quantities.add('cosmological_redshift')


    def _iter_native_dataset(self, pre_filters=None):
        with h5py.File(self._file, 'r') as fh:
            for key in fh:
                if key == 'cosmology':
                    continue
                d = fh[key]
                z = d.attrs['z']
                if pre_filters is None or all(f[0](*([z]*(len(f)-1))) for f in pre_filters):
                    yield d


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        if native_quantity == 'cosmological_redshift':
            data = np.empty(dataset['redshift'].shape)
            data.fill(dataset.attrs['z'])
            return data
        return dataset[native_quantity].value
