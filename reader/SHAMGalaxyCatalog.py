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
        if match_to not in ('LiWhite', 'MBII'):
            raise ValueError('`match_to` must be "LiWhite" or "MBII"')
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
        self._distmod = self.cosmology.distmod(self.redshift).value
        self.box_size = (100.0/self._h)
        self.overdensity = 97.7
        self.lightcone = False
        self.SDSS_kcorrection_z = 0.0

        self.quantities  = { 'redshift':         ('redshift', None),
                             'stellar_mass':     ('sm', None),
                             'halo_id':          ('id', None),
                             'parent_halo_id':   ('upid', None),
                             'positionX':        ('x', lambda x: x/self._h),
                             'positionY':        ('y', lambda x: x/self._h),
                             'positionZ':        ('z', lambda x: x/self._h),
                             'velocityX':        ('vx', None),
                             'velocityY':        ('vy', None),
                             'velocityZ':        ('vz', None),
                             'mass':             ('mvir', lambda x: x/self._h),
                             'SDSS_u:observed:': ('AMAG[0]', lambda x: x+self._distmod),
                             'SDSS_g:observed:': ('AMAG[1]', lambda x: x+self._distmod),
                             'SDSS_r:observed:': ('AMAG[2]', lambda x: x+self._distmod),
                             'SDSS_i:observed:': ('AMAG[3]', lambda x: x+self._distmod),
                             'SDSS_z:observed:': ('AMAG[4]', lambda x: x+self._distmod),
                             'SDSS_u:rest:':     ('AMAG[0]', None),
                             'SDSS_g:rest:':     ('AMAG[1]', None),
                             'SDSS_r:rest:':     ('AMAG[2]', None),
                             'SDSS_i:rest:':     ('AMAG[3]', None),
                             'SDSS_z:rest:':     ('AMAG[4]', None),
                           }
        
        
    def get_quantities(self, quantities, filters={}):
        if isinstance(quantities, basestring):
            quantities = [quantities]
        
        if not quantities:
            raise ValueError('quantities cannot be empty')
        
        if not all(q in self.quantities for q in quantities):
            raise ValueError('Some quantities are not available in this catalog')

        if self.redshift < filters.get('zlo', -np.inf) or self.redshift > filters.get('zhi', np.inf):
            result = [np.array([]) for _ in quantities]
        else:
            result = []
            for q in quantities:
                if q in self.data_cache:
                    result.append(self.data_cache[q])
                else:
                    key, func = self.quantities[q]
                    d = func(self.npz_file[key]) if callable(func) else self.npz_file[key]
                    self.data_cache[q] = d
                    result.append(d)
        
        return result if len(result) > 1 else result[0]

