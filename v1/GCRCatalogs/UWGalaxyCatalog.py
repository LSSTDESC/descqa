# University of Washington galaxy catalog class.

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
import astropy.cosmology
import astropy.units as u

class UWGalaxyCatalog(GalaxyCatalog):
    """
    University of Washington galaxy catalog class. Uses generic quantity and
    filter mechanisms defined by GalaxyCatalog base class. In addition,
    implements the use of 'stored' vs. 'derived' quantity getter methods.
    Additional data structures:

    catalog       A dictionary whose keys are the names of the various stored
                  properties, and whose values are arrays containing the values
                  of these quantities for each of the galaxies in the catalog.

    Ngals         The number of galaxies in the catalog.

    derived       A dictionary whose keys are the names of derived quantities
                  and whose values are tuples containing the string name of a
                  corresponding stored quantity (actually present in the file)
                  and a pointer to the function used to compute the derived
                  quantity from the stored one. Some catalogs may support
                  having the stored quantity be a tuple of stored quantity
                  names.
    """


    def __init__(self, **kwargs):
        fn = kwargs.get('fn')
        self.type_ext     = 'uw'
        self.filters      = { 'zlo':                True,
                              'zhi':                True
                            }
        self.quantities   = { 'redshift':           self._get_stored_property,
                              'stellar_mass':       self._get_derived_property
                            }
        self.derived      = { 'stellar_mass':       (('mass_stellar', 1.e10), self._multiply)
                            }
        self.catalog      = {}
        self.Ngals        = 0
        self.sky_area     = 4.*np.pi*u.sr   # all sky by default
        self.cosmology    = None
        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        self.catalog = np.genfromtxt(fn, delimiter=',', names=True)
        self.Ngals = self.catalog.shape[0]  # TODO: CHECK THIS for number of galaxies
        # TODO: how to get cosmology?
        # TODO: how to get sky area?
        return self

    # Functions for applying filters

    def _construct_mask(self, filters):
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

    def _get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        filter_mask = self._construct_mask(filters)
        return self.catalog[quantity][np.where(filter_mask)]

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
            values = self.catalog[stored_qty_name[0]][np.where(filter_mask)]
            return stored_qty_fctn(values, stored_qty_name[1:])
        else:
            values = self.catalog[stored_qty_name][np.where(filter_mask)]
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
