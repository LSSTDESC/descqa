# SHAM galaxy catalog class
# Contact: Yao-Yuan Mao <yymao.astro@gmail.com>

import os
import numpy as np
from GalaxyCatalogInterface import GalaxyCatalog
from astropy.cosmology import FlatLambdaCDM

class _FunctionWrapper:
    def __init__(self, d, k):
        self._d = d
        self._k = k

    def __call__(self, quantity, filters):
        return self._d[self._k]

class SHAMGalaxyCatalog(GalaxyCatalog):
    """
    SHAM galaxy catalog class.
    """

    def __init__(self, fn=None):
        self.type_ext = 'npy'
        self.redshift = 0.062496
        self.cosmology = FlatLambdaCDM(H0=70.2, Om0=0.275, Ob0=0.046)
        self._h = self.cosmology.H0.value / 100.0
        self.box_size = (100.0/self._h)
        self.overdensity = 97.7
        self.lightcone = False
        self._data = {}
        self.quantities  = { 'stellar_mass':   _FunctionWrapper(self._data, 'sm'),
                             'halo_id':        _FunctionWrapper(self._data, 'id'),
                             'parent_halo_id': _FunctionWrapper(self._data, 'upid'),
                             'positionX':      _FunctionWrapper(self._data, 'x'),
                             'positionY':      _FunctionWrapper(self._data, 'y'),
                             'positionZ':      _FunctionWrapper(self._data, 'z'),
                             'velocityX':      _FunctionWrapper(self._data, 'vx'),
                             'velocityY':      _FunctionWrapper(self._data, 'vy'),
                             'velocityZ':      _FunctionWrapper(self._data, 'vz'),
                             'mass':           _FunctionWrapper(self._data, 'mvir'),
                           }

        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        cat = np.load(fn)
        
        halos = np.load(os.path.join(os.path.dirname(fn), 'hlist_{0:.5f}.npy'.format(1.0/(1.0+self.redshift))))
        s = halos.argsort(order='id')
        halos = halos[s[np.searchsorted(halos['id'], cat['id'], sorter=s)]]
        del s
        assert (halos['id'] == cat['id']).all()

        for name in cat.dtype.names:
            self._data[name] = cat[name]

        for name in halos.dtype.names:
            if name == 'id':
                continue
            self._data[name] = halos[name]
        
        for name in ('x', 'y', 'z', 'mvir'):
            self._data[name] /= self._h

        return self

