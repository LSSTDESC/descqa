# SHAM galaxy catalog class
# Contact: Yao-Yuan Mao <yymao@stanford.edu>

import os
import numpy as np
from GalaxyCatalogInterface import GalaxyCatalog


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
        self.root_path= '/global/project/projectdirs/lsst/descqa/catalog'
        self.redshift = 0.062496
        self.box_size = 100.0
        self.lightcone = False
        self._data = {}
        self.quantities  = { 'stellar_mass': _FunctionWrapper(self._data, 'sm'),
                             'halo_id':      _FunctionWrapper(self._data, 'id'),
                             'positionX':    _FunctionWrapper(self._data, 'x'),
                             'positionY':    _FunctionWrapper(self._data, 'y'),
                             'positionZ':    _FunctionWrapper(self._data, 'z'),
                             'velocityX':    _FunctionWrapper(self._data, 'vx'),
                             'velocityY':    _FunctionWrapper(self._data, 'vy'),
                             'velocityZ':    _FunctionWrapper(self._data, 'vz'),
                             'mass':         _FunctionWrapper(self._data, 'mvir'),
                           }

        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        if fn is None:
            fn = os.path.join(self.root_path, 'SHAM_{0:.5f}.npy'.format(1.0/(1.0+self.redshift)))
        cat = np.load(fn)

        halos = np.load(os.path.join(self.root_path, 'MBII-DMO', 'hlist_{0:.5f}.npy'.format(1.0/(1.0+self.redshift))))
        s = halos.argsort(order='id')
        halos = halos[s[np.searchsorted(halos['id'], cat['id'], sorter=s)]]
        assert (halos['id'] == cat['id']).all()

        for name in cat.dtype.names:
            self._data[name] = cat[name]

        for name in halos.dtype.names:
            if name == 'id':
                continue
            self._data[name] = halos[name]

        return self

