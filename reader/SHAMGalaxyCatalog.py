# SHAM galaxy catalog class
# Contact: Yao-Yuan Mao <yymao.astro@gmail.com>

import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
#from GalaxyCatalogInterface import GalaxyCatalog
GalaxyCatalog = object

class SHAMGalaxyCatalog(GalaxyCatalog):
    """
    SHAM galaxy catalog class.
    """

    def __init__(self, redshift=0.062496, match_to='LiWhite', **kwargs):
        if match_to not in ('LiWhite', 'MB2'):
            raise ValueError('`match_to` must be "LiWhite" or "MB2"')
        self.match_to = match_to
        self.redshift = redshift
        self.scale    = 1.0/(1.0+self.redshift)

        self.base_catalog_dir = kwargs['base_catalog_dir']
        self.filename = os.path.join(self.base_catalog_dir, 'SHAM_{:.5f}_{}.npz'.format(self.scale, self.match_to))
        if not os.path.isfile(self.filename):
            raise ValueError('{} does not exist!'.format(self.filename))
        self.npz_file = np.load(self.filename)
        self.data_cache = {}

        self.cosmology = FlatLambdaCDM(H0=70.2, Om0=0.275, Ob0=0.046)
        self._h = self.cosmology.H0.value / 100.0
        self.box_size = (100.0/self._h)
        self.overdensity = 97.7
        self.lightcone = False

        self.quantities  = { 'stellar_mass':   'sm',
                             'halo_id':        'id',
                             'parent_halo_id': 'upid',
                             'positionX':      'x',
                             'positionY':      'y',
                             'positionZ':      'z',
                             'velocityX':      'vx',
                             'velocityY':      'vy',
                             'velocityZ':      'vz',
                             'mass':           'mvir',
                           }
        
        if self.match_to == 'LiWhite':
            self.quantities['SDSS_g:observed:'] = 'g_mag'
            self.quantities['SDSS_r:observed:'] = 'r_mag'
            self.quantities['SDSS_i:observed:'] = 'i_mag'
            self.quantities['SDSS_z:observed:'] = 'z_mag'

        
    def get_quantities(self, quantities, filters={}):
        if isinstance(quantities, basestring):
            quantities = [quantities]
        
        if not quantities:
            raise ValueError('quantities cannot be empty')

        result = []
        for q in quantities:
            if q in self.data_cache:
                result.append(self.data_cache[q])
            elif q in self.quantities:
                key = self.quantities[q]
                d = self.npz_file[key]
                if key in ('x', 'y', 'z', 'mvir'):
                    d /= self._h
                self.data_cache[q] = d
                result.append(d)
            else:
                raise ValueError('{} not available'.format(q))
        
        return result if len(result) > 1 else result[0]

