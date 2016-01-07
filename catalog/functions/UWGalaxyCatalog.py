# University of Washington galaxy catalog class.

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
import astropy.cosmology

class UWGalaxyCatalog(GalaxyCatalog):

    def __init__(self, fn=None):
        self.type_ext     = 'uw'
        self.filters      = { 'zlo':                True,
                              'zhi':                True
                            }
        self.quantities   = { 'redshift':           self.get_stored_property,
                              'stellar_mass':       self.get_derived_property
                            }
        self.derived      = { 'stellar_mass':       (('mass_stellar', 1.e10), self.multiply)
                            }
        self.catalog      = {}
        self.Ngals        = 0
        self.cosmology    = None
        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        self.catalog = np.genfromtxt(fn, delimiter=',', names=True)
        self.Ngals = self.catalog.shape[0]  # CHECK THIS for number of galaxies
        # how to get cosmology?
        return self

    # Functions for applying filters

    def construct_mask(self, filters):
        """
        Given a dictionary of filter constraints, construct a mask array
        for use in filtering the catalog.
        """
        if type(filters) is not dict:
            raise TypeError("construct_mask: filters must be given as dict")
        mask = np.ones((self.Ngals), dtype=bool_)
        for filter_name in filters.keys():
            if filter_name == 'zlo':
                mask = mask & (filters[filter_name] < self.catalog['redshift'])
            elif filter_name == 'zhi':
                mask = mask & (filters[filter_name] > self.catalog['redshift'])
        return mask

    # Functions for returning quantities from the catalog

    def get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        filter_mask = self.construct_mask(filters)
        return self.catalog[quantity][np.where(filter_mask)]

    def get_derived_property(self, quantity, filters):
        """
        Return a derived halo property. These properties aren't stored
        in the catalog but can be computed from properties that are via
        a simple function call.
        """
        filter_mask = self.construct_mask(filters)
        stored_qty_rec = self.derived[quantity]
        stored_qty_name = stored_qty_rec[0]
        stored_qty_fctn = stored_qty_rec[1]
        if type(stored_qty_name) is tuple:
            values = self.catalog[stored_qty_name[0]][np.where(filter_mask)]
            return stored_qty_fctn(values, stored_qty_name[1:])
        else:
            values = self.catalog[stored_qty_name][np.where(filter_mask)]
            return stored_qty_fctn(values)

    # Functions for computing derived values

    def translate(self, propList):
        """
        Translation routine -- a passthrough that accomplishes mapping of
        derived quantity names to stored quantity names via the derived
        property function mechanism.
        """
        return propList

    def multiply(self, propList, factor_tuple):
        """
        Multiplication routine -- derived quantity is equal to a stored
        quantity times some factor. Additional args for the derived quantity
        routines are passed in as a tuple, so extract the factor first.
        """
        factor = factor_tuple[0]
        return propList * factor
