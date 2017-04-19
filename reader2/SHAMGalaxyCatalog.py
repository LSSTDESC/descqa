"""
SHAM galaxy catalog class
Contact: Yao-Yuan Mao <yymao.astro@gmail.com>
"""
from __future__ import division
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from .BaseGalaxyCatalog import BaseGalaxyCatalog

__all__ = ['SHAMGalaxyCatalog']

class SHAMGalaxyCatalog(BaseGalaxyCatalog):
    """
    SHAM galaxy catalog class.
    """

    def _subclass_init(self, redshift=0.062496, match_to='LiWhite', base_catalog_dir=os.curdir, **kwargs):
        if match_to not in ('LiWhite', 'MBII'):
            raise ValueError('`match_to` must be "LiWhite" or "MBII"')
        self.match_to = match_to
        self.redshift = redshift
        self.scale = 1.0/(1.0+self.redshift)

        self.base_catalog_dir = base_catalog_dir
        self.filename = os.path.join(self.base_catalog_dir, 'SHAM_{:.5f}_{}.npz'.format(self.scale, self.match_to))
        if not os.path.isfile(self.filename):
            raise ValueError('{} does not exist!'.format(self.filename))
        self.npz_file = np.load(self.filename)

        self.cosmology = FlatLambdaCDM(H0=70.2, Om0=0.275, Ob0=0.046)
        self._h = self.cosmology.H0.value / 100.0
        self._distmod = self.cosmology.distmod(self.redshift).value
        self.box_size = (100.0/self._h)
        self.overdensity = 97.7
        self.lightcone = False
        self.SDSS_kcorrection_z = 0.0

        self._quantity_modifiers = {
            'stellar_mass':     'sm',
            'halo_id':          'id',
            'parent_halo_id':   'upid',
            'positionX':        (lambda x: x/self._h, 'x'),
            'positionY':        (lambda x: x/self._h, 'y'),
            'positionZ':        (lambda x: x/self._h, 'z'),
            'velocityX':        'vx',
            'velocityY':        'vy',
            'velocityZ':        'vz',
            'mass':             (lambda x: x/self._h, 'mvir'),
            'LSST_u:observed:': (lambda x: x+self._distmod, 'AMAG[0]'),
            'LSST_g:observed:': (lambda x: x+self._distmod, 'AMAG[1]'),
            'LSST_r:observed:': (lambda x: x+self._distmod, 'AMAG[2]'),
            'LSST_i:observed:': (lambda x: x+self._distmod, 'AMAG[3]'),
            'LSST_z:observed:': (lambda x: x+self._distmod, 'AMAG[4]'),
            'LSST_u:rest:':     'AMAG[0]',
            'LSST_g:rest:':     'AMAG[1]',
            'LSST_r:rest:':     'AMAG[2]',
            'LSST_i:rest:':     'AMAG[3]',
            'LSST_z:rest:':     'AMAG[4]',
        }

        self._native_quantities = set(self.npz_file.keys())

        self._pre_filter_quantities = {'redshift'}


    def _iter_native_dataset(self, pre_filters=None):
        zfliters = [f[0] for f in pre_filters if len(f) == 2 and f[1] == 'redshift'] if pre_filters else None
        if all(zfliter(self.redshift) for zfilter in zfilters):
            yield self.npz_file


    @staticmethod
    def _fetch_native_quantity(dataset, native_quantity):
        return dataset[native_quantity]
