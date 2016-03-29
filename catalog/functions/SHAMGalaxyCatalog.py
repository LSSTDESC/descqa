# SHAM galaxy catalog class
# Contact: Yao-Yuan Mao <yymao@stanford.edu>

import os
import numpy as np
from GalaxyCatalogInterface import GalaxyCatalog
import astropy.cosmology

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
        self.quantities  = { 'stellar_mass': self._stored_quantity_wrapper('sm'),
                             'halo_id':      self._stored_quantity_wrapper('id'),
                             'positionX':    self._stored_quantity_wrapper('x'),
                             'positionY':    self._stored_quantity_wrapper('y'),
                             'positionZ':    self._stored_quantity_wrapper('z'),
                           }

        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn=None):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        if fn is None:
            fn = os.path.join(self.root_path, 'SHAM_{0:.5f}.npy'.format(1.0/(1.0+self.redshift)))
        self._data = np.load(fn)

        print "WARNING: Initializing cosmology using built-in astropy cosmology LambdaCDM"
        self.cosmology = astropy.cosmology.LambdaCDM(H0   = 70.2,
                Om0  = 0.275,
                Ode0 = 0.725)
         
        return self

    def _stored_quantity_wrapper(self, name):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        return (lambda quantity, filters: self._data[name])
