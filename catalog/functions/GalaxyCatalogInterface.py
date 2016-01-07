# DESCQA galaxy catalog interface. This defines the GalaxyCatalog base class
# and, on import, registers all of the available catalog readers. Convenience
# functions are defined that enable automatic detection of the appropriate
# catalog type.

# Note: right now we are working with galaxy properties as floats, with
# expected return units listed in GalaxyCatalog.__init__ below. In the future
# we might move to expecting values as Astropy Quantity objects.

import glob
import importlib
import os.path
from descqaGlobalConfig import *
import numpy as np
import astropy.units as u

# Galaxy catalog base class.

class GalaxyCatalog(object):
    """
    Base class for galaxy catalog classes. Common internal data structures:

    type_ext      A string giving the file name extension, for catalogs that use
                  the default method for determining file type.

    filters       A dictionary whose keys are strings giving the names of
                  filters supported by the catalog class, and whose values are
                  the methods used to apply these constraints (or True if they
                  are supported but handled via a different mechanism). The
                  default implementation sets this dictionary to include keys
                  that should be supported by all catalogs.

    quantities    A dictionary whose keys are strings giving the names of
                  quantities that can be requested from the catalog, and whose
                  values are the methods used to request these quantities. The
                  methods should take two arguments: the name of the quantity
                  and a dictionary containing the filters to be applied and the
                  values for the filters. The default implementation sets this
                  dictionary to include keys that should be supported by all
                  catalogs.

    sky_area      The sky area covered by the catalog as an Astropy Quantity
                  object.

    cosmology     Should be set by load routines to an Astropy.cosmology object
                  encoding the cosmology used to generate the catalog. This
                  allows calling programs to compute things like comoving
                  volumes appropriately. None by default.
    """

    type_ext    = ''
    filters     = {'zlo'                  : None,    # min redshift
                   'zhi'                  : None     # max redshift
                  }
    quantities  = {'stellar_mass'         : None     # stellar mass in M_sun
                  }
    sky_area    = 4.*np.pi*u.sr   # all sky by default
    cosmology   = None

    def __init__(self, fn=None):
        """
        Default GalaxyCatalog constructor takes one optional filename argument.
        If present, the referenced catalog is checked for validity and loaded
        if possible. If it is not valid, a ValueError is raised. If no argument
        is given, an instance of the class is created without internal data.
        Subclass __init__ methods that override this one should call this one
        just before they return.
        """
        if fn:
            if self.is_valid(fn):
                self.load(fn)
            else:
                raise ValueError('invalid catalog file')

    def is_valid(self, fn):
        """
        Given a catalog path, determine whether it is a valid catalog of this
        type. The default implementation merely checks the filename extension
        against the type_ext attribute of the class.
        """
        base = os.path.basename(fn)
        ext = base.split('.')[-1]
        return (ext == self.type_ext)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures. Should return self if successful.
        """
        return self

    def get_cosmology(self):
        """
        Return as an Astropy.cosmology object the cosmological parameter values
        assumed in generating this catalog.
        """
        return self.cosmology

    def get_sky_area(self):
        """
        Return the sky area covered by the catalog as an Astropy Quantity.
        """
        return self.sky_area

    def get_quantities(self, ids, filters):
        """
        Given a list of string property names and optional filter arguments,
        return as a list of NumPy arrays the selected values from the catalog.
        A single property name can also be passed, in which case the result
        is a single NumPy array. Filters are specified using a dictionary in
        which the keys are string constraint names and the values are the
        constraints.
        """
        if type(ids) is list:
            idList = ids
        elif type(ids) is str:
            idList = [ids]
        else:
            raise TypeError("get_quantities: ids must be list or str")
        if type(filters) != dict:
            raise TypeError("get_quantities: filters must be dict")

        okQuantities = self.get_supp_quantities()
        for quantity in idList:
            if quantity not in okQuantities:
                raise ValueError("get_quantities: quantity '%s' not supported" % quantity)
        okFilters = self.get_supp_filters()
        for filt in filters.keys():
            if filt not in okFilters:
                raise ValueError("get_quantities: filter '%s' not supported" % filt)

        results = []
        for quantity in idList:
            quantityGetter = self.quantities[quantity]
            results.append(quantityGetter(quantity, filters))
        if type(ids) == list:
            return results
        else:
            return results[0]

    def get_supp_filters(self):
        """
        Return a list containing the supported filter keywords for this
        catalog.
        """
        return self.filters.keys()
        filterList   = self.filters.keys()
        filterListOK = []
        for filt in filterList:
            if self.filters[filt]:
                filterListOK.append(filt)
        return filterListOK

    def get_supp_quantities(self):
        """
        Return a list containing the supported quantities for this
        catalog.
        """
        quantityList   = self.quantities.keys()
        quantityListOK = []
        for quantity in quantityList:
            if self.quantities[quantity]:
                quantityListOK.append(quantity)
        return quantityListOK

# Convenience function for loading generic galaxy catalogs from a file.

def loadCatalog(fn):
    """
    Convenience function to enable loading of generic galaxy catalogs. Each of
    the registered types is tried in turn, and the load method of the first
    match is invoked, returning the catalog object. If the given path is not
    accessible or no match is found, None is returned.
    """
    if os.path.exists(fn):
        for catalogType in catalog_class_registry.keys():
            catalogObject = catalog_class_registry[catalogType]()
            if catalogObject.is_valid(fn):
                print('file %s is of type %s.' % (fn, catalogType))
                return catalogObject.load(fn)
            else:
                del catalogObject
        # only get here if no catalog types matched
        print('unknown catalog type')
        return None
    else:
        print('catalog not accessible')
        return None

# Search the Python path for any galaxy catalog modules and import them.

catalog_class_registry = {}

files = glob.glob(os.path.join(DESCQACatalogFunctionDir, '*GalaxyCatalog.py'))

for this_file in files:
    print('importing %s' % this_file)
    class_name = os.path.basename(this_file).split('.')[0]
    module = importlib.import_module(class_name)
    catalog_class_registry[class_name] = getattr(module, class_name)
