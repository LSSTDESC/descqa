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

        self._pre_filter_quantities = {'redshift'}

        self._quantity_modifiers = {
            'mass':         (lambda x: x**10.0, 'log_halomass'),
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


    def _iter_native_dataset(self, pre_filters=None):
        zfliters = [f[0] for f in pre_filters if len(f) == 2 and f[1] == 'redshift'] if pre_filters else None

        maxdz = 0.01345
        with h5py.File(self._file, 'r') as fh:
            for key in fh:
                if key == 'cosmology':
                    continue
                d = fh[key]
                z_test = np.linspace(d.attrs['z']-maxdz, d.attrs['z']+maxdz, 100)
                if all(zfliter(z_test).any() for zfliter in zfliters):
                    yield d


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        return dataset[native_quantity].value
