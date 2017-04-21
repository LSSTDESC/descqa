# Massive Black 2 galaxy catalog class

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
import astropy.cosmology
import h5py
import astropy.units as u

class iHODGalaxyCatalog(GalaxyCatalog):
    """
    iHOD galaxy catalog class.
    """

    def __init__(self, **kwargs):
        fn = kwargs.get('fn')
        self.type_ext =   'iHOD'
        self.filters  = {
                          'zlo':                   True,
                          'zhi':                   True
                        }
        self.quantities = {
                             'positionX':                 self._get_stored_property,
                             'positionY':                 self._get_stored_property,
                             'positionZ':                 self._get_stored_property,
                             'velocityX':                 self._get_stored_property,
                             'velocityY':                 self._get_stored_property,
                             'velocityZ':                 self._get_stored_property,
                             'stellar_mass':              self._get_stored_property,
                             'mass':                      self._get_stored_property,
                             'halo_id':                   self._get_stored_property,
                             'parent_halo_id':            self._get_stored_property,
                             'SDSS_g:rest:':          self._get_stored_property,
                             'SDSS_r:rest:':          self._get_stored_property,
                           }

        self.Ngals        = 0
        self.sky_area     = 4.*np.pi*u.sr   # all sky by default
        self.cosmology    = None
        self.lightcone    = False
        self.box_size     = 100.0 / 0.701
        return GalaxyCatalog.__init__(self, fn)

    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """
        self.catalog = self._read_rec_from_hdf5(fn, group='galaxy')   
        # turam: Add placeholder redshift; confirm correctness (YZ: good)
        self.redshift = (1.0 / 0.941176) - 1.0
        # turam: Confirm cosmology is correct (YZ: good)
        self.cosmology = astropy.cosmology.FlatLambdaCDM(H0=70.1, Om0=0.275, Ob0=0.046)

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
        for filter_name in filters.keys():
            if filter_name == 'zlo':
                mask = mask & (filters[filter_name] < self.redshift)
            elif filter_name == 'zhi':
                mask = mask & (filters[filter_name] > self.redshift)
        return mask

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

    def _read_rec_from_hdf5(self, h5file, group="galaxy"):
        recdict = self._read_recdict_from_hdf5(h5file)
        return(recdict[group])

    def _read_recdict_from_hdf5(self, h5file):
        """ read catalog as a dictionary of record arrays.
        """
        f = h5py.File(h5file, "r")
        recdict = {}
        for grp, val in f.iteritems():
            print grp
            datasets = []
            dtypes = []
            for key in f[grp].keys():
                dset = f[grp][key][:]
                dtypename = f[grp][key].dtype.name
                dtype = (str(key), dtypename)
                datasets.append(dset)
                dtypes.append(dtype)
            recdict[str(grp)] = np.rec.fromarrays(tuple(datasets), dtype=dtypes)
        f.close()
        return(recdict)

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

    def _power(self, propList, base_tuple):
        return  base_tuple[0] ** propList
