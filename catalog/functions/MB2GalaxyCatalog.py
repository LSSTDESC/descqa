# Massive Black 2 galaxy catalog class

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
from astropy.table import Table
import astropy.units as u

class MB2GalaxyCatalog(GalaxyCatalog):
    """
    Massive Black 2 galaxy catalog class.
    """

    def __init__(self, fn=None):
        self.type_ext =   'MB2'
        self.filters  = {
                          'zlo':                   True,
                          'zhi':                   True
                        }
        self.h          = 0.702
        self.quantities = {
                             'redshift':              self._get_stored_property,
                             'positionX':             self._get_derived_property,  # Position returned in Mpc, stored in kpc/h 
                             'positionY':             self._get_derived_property,
                             'positionZ':             self._get_derived_property,
                             'velocityZ':             self._get_stored_property,   # Velocity returned in km/sec
                             'mass':                  self._get_derived_property,  # Masses returned in Msun but stored in 1e10 Msun/h
                             'stellar_mass':          self._get_derived_property,
                             'gas_mass':              self._get_stored_property,
                             'sfr':                   self._get_stored_property,
                             'SDSS_u:rest:':          self._get_stored_property,    # don't have a way to return these yet
                             'SDSS_g:rest:':          self._get_stored_property,
                             'SDSS_r:rest:':          self._get_stored_property,
                             'SDSS_i:rest:':          self._get_stored_property,
                             'SDSS_z:rest:':          self._get_stored_property,
                           }

        self.derived      = {
                             'mass':            (('mass', self.h * 1.e10), self._multiply),
                             'stellar_mass':    (('stellar_mass', self.h * 1.e10), self._multiply),
                             'positionX':       (('x', self.h * 1.e-3), self._multiply), # Position stored in kpc/h
                             'positionY':       (('y', self.h * 1.e-3), self._multiply),
                             'positionZ':       (('z', self.h * 1.e-3), self._multiply), 
                            }
        self.Ngals        = 0
        self.sky_area     = 4.*np.pi*u.sr   # all sky by default
        self.lightcone    = False
        self.box_size     = 100.0
        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        self.catalog = Table.read(fn, path='data')
        self.Ngals = len(self.catalog)

        return self

    def _construct_mask(self, filters):
        """
        Given a dictionary of filter constraints, construct a mask array
        for use in filtering the catalog.
        """
        if type(filters) is not dict:
            raise TypeError("construct_mask: filters must be given as dict")
        mask = np.ones((self.Ngals), dtype=np.bool_)
        mask = mask & (np.isfinite(self.catalog['x'])) # filter out NaN positions from catalog
        mask = mask & (np.isfinite(self.catalog['y']))
        mask = mask & (np.isfinite(self.catalog['z']))
        for filter_name in filters.keys():
            if filter_name == 'zlo':
                mask = mask & (filters[filter_name] < self.catalog['redshift'])
            elif filter_name == 'zhi':
                mask = mask & (filters[filter_name] > self.catalog['redshift'])
        return mask

    def _get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        filter_mask = self._construct_mask(filters)
        return self.catalog[quantity][np.where(filter_mask)].data

    def _get_derived_property(self, quantity, filters):
        """
        Return a derived halo property. These properties aren't stored
        in the catalog but can be computed from properties that are via
        a simple function call.
        """
        filter_mask = self._construct_mask(filters)
        stored_qty_rec = self.derived[quantity]
        stored_qty_name = stored_qty_rec[0]
        stored_qty_fctn = stored_qty_rec[1]
        if type(stored_qty_name) is tuple:
            values = self.catalog[stored_qty_name[0]][np.where(filter_mask)].data
            return stored_qty_fctn(values, stored_qty_name[1:])
        else:
            values = self.catalog[stored_qty_name][np.where(filter_mask)].data
            return stored_qty_fctn(values)

    # Functions for computing derived values
    def _translate(self, propList):
        """
        Translation routine -- a passthrough that accomplishes mapping of
        derived quantity names to stored quantity names via the derived
        property function mechanism.
        """
        return propList

    def _multiply(self, propList, factor_tuple):
        """
        Multiplication routine -- derived quantity is equal to a stored
        quantity times some factor. Additional args for the derived quantity
        routines are passed in as a tuple, so extract the factor first.
        """
        factor = factor_tuple[0]
        return propList * factor
