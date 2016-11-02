# Argonne galaxy catalog class.

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
import h5py
import astropy.cosmology
import astropy.units as u

class GalacticusGalaxyCatalog(GalaxyCatalog):
    """
    Argonne galaxy catalog class. Uses generic quantity and filter mechanisms
    defined by GalaxyCatalog base class. In addition, implements the use of
    'stored' vs. 'derived' quantity getter methods. Additional data structures:

    catalog       A dictionary whose keys are halo names and whose values are
                  themselves dictionaries. The halo dictionaries have as keys
                  the names of the various stored properties, and as values
                  arrays containing the values of these quantities for each of
                  the galaxies in the halos.

    derived       A dictionary whose keys are the names of derived quantities
                  and whose values are tuples containing the string name of a
                  corresponding stored quantity (actually present in the file)
                  and a pointer to the function used to compute the derived
                  quantity from the stored one. Some catalogs may support
                  having the stored quantity be a tuple of stored quantity
                  names.
    """

    def __init__(self, fn=None):
        self.type_ext    = 'galacticus'
        self.filters     = { 'zlo':                   True,
                             'zhi':                   True
                           }
        self.quantities  = { 'redshift':              self._get_stored_property,
                             'ra':                    self._get_stored_property,
                             'dec':                   self._get_stored_property,
                             'v_pec':                 self._get_stored_property,
                             'mass':                  self._get_derived_property,
                             'age':                   self._get_stored_property,
                             'stellar_mass':          self._get_derived_property,
                             'log_stellarmass':       self._get_stored_property,
                             'log_halomass':          self._get_stored_property,
                             'gas_mass':              self._get_stored_property,
                             'metallicity':           self._get_stored_property,
                             'sfr':                   self._get_stored_property,
                             'ellipticity':           self._get_stored_property,
                             #'positionX':             self._get_derived_property, #units are in physical Mpc
                             #'positionY':             self._get_derived_property,
                             #'positionZ':             self._get_derived_property,
                             'positionX':             self._get_stored_property, #units are now in comoving Mpc
                             'positionY':             self._get_stored_property,
                             'positionZ':             self._get_stored_property,
                             'velocityX':             self._get_stored_property,
                             'velocityY':             self._get_stored_property,
                             'velocityZ':             self._get_stored_property,
                             'disk_ra':               self._get_stored_property,
                             'disk_dec':              self._get_stored_property,
                             'disk_sigma0':           self._get_stored_property,
                             'disk_re':               self._get_stored_property,
                             'disk_index':            self._get_stored_property,
                             'disk_a':                self._get_stored_property,
                             'disk_b':                self._get_stored_property,
                             'disk_theta_los':        self._get_stored_property,
                             'disk_phi':              self._get_stored_property,
                             'disk_stellarmass':      self._get_derived_property,
                             'log_disk_stellarmass':  self._get_stored_property,
                             'disk_metallicity':      self._get_stored_property,
                             'disk_age':              self._get_stored_property,
                             'disk_sfr':              self._get_stored_property,
                             'disk_ellipticity':      self._get_stored_property,
                             'bulge_ra':              self._get_stored_property,
                             'bulge_dec':             self._get_stored_property,
                             'bulge_sigma0':          self._get_stored_property,
                             'bulge_re':              self._get_stored_property,
                             'bulge_index':           self._get_stored_property,
                             'bulge_a':               self._get_stored_property,
                             'bulge_b':               self._get_stored_property,
                             'bulge_theta_los':       self._get_stored_property,
                             'bulge_phi':             self._get_stored_property,
                             'bulge_stellarmass':     self._get_derived_property,
                             'log_bulge_stellarmass': self._get_stored_property,
                             'bulge_age':             self._get_stored_property,
                             'bulge_sfr':             self._get_stored_property,
                             'bulge_metallicity':     self._get_stored_property,
                             'bulge_ellipticity':     self._get_stored_property,
                             'agn_ra':                self._get_stored_property,
                             'agn_dec':               self._get_stored_property,
                             'agn_mass':              self._get_stored_property,
                             'agn_accretnrate':       self._get_stored_property,
                             'SDSS_u:rest:':          self._get_stored_property,
                             'SDSS_g:rest:':          self._get_stored_property,
                             'SDSS_r:rest:':          self._get_stored_property,
                             'SDSS_i:rest:':          self._get_stored_property,
                             'SDSS_z:rest:':          self._get_stored_property,
                             'SDSS_u:observed:':      self._get_stored_property,
                             'SDSS_g:observed:':      self._get_stored_property,
                             'SDSS_r:observed:':      self._get_stored_property,
                             'SDSS_i:observed:':      self._get_stored_property,
                             'SDSS_z:observed:':      self._get_stored_property,
                             'invscalefactor':        self._get_derived_property,
                             'DES_g:rest:':           None,
                             'DES_r:rest:':           None,
                             'DES_i:rest:':           None,
                             'DES_z:rest:':           None,
                             'DES_Y:rest:':           None,
                             'DES_g:observed:':       None,
                             'DES_r:observed:':       None,
                             'DES_i:observed:':       None,
                             'DES_z:observed:':       None,
                             'DES_Y:observed:':       None,
                             'LSST_u:rest:':          None,
                             'LSST_g:rest:':          None,
                             'LSST_r:rest:':          None,
                             'LSST_i:rest:':          None,
                             'LSST_z:rest:':          None,
                             'LSST_y4:rest:':         None,
                             'LSST_u:observed:':      None,
                             'LSST_g:observed:':      None,
                             'LSST_r:observed:':      None,
                             'LSST_i:observed:':      None,
                             'LSST_z:observed:':      None,
                             'LSST_y4:observed:':     None,
                             'B':                     None,
                             'U':                     None,
                             'V':                     None,
                             'CFHTL_g:rest:':         None,
                             'CFHTL_r:rest:':         None,
                             'CFHTL_i:rest:':         None,
                             'CFHTL_z:rest:':         None,
                             'CFHTL_g:observed:':     None,
                             'CFHTL_r:observed:':     None,
                             'CFHTL_i:observed:':     None,
                             'CFHTL_z:observed:':     None,
                           }
        self.derived     = {'stellar_mass':      ('log_stellarmass',       self._unlog10),
                            'mass':              ('log_halomass',  self._unlog10),
                            'disk_stellarmass':  ('log_disk_stellarmass',  self._unlog10),
                            'bulge_stellarmass': ('log_bulge_stellarmass', self._unlog10)}
                            #'positionX'        : (('positionX',1.05),self._multiply),  #convert to comoving *1/scale-factor
                            #'positionY'        : (('positionY',1.05),self._multiply),
                            #'positionZ'        : (('positionZ',1.05),self._multiply)}
        
        self.catalog     = {}
        self.sky_area    = 4.*np.pi*u.sr   # all sky by default
        self.cosmology   = None
        self.lightcone   = False
        self.box_size    = None
        return GalaxyCatalog.__init__(self, fn)


    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """

        hdfFile = h5py.File(fn, 'r')
        hdfKeys, hdfAttrs = self._gethdf5group(hdfFile)
        print "load using keys: ", hdfKeys
        self.catalog = {}
        for key in hdfKeys:
            #if 'Output' in key:
            if 'Output19' in key:
                outgroup = hdfFile[key]
                dataKeys, dataAttrs = self._gethdf5group(outgroup)
                self.catalog[key] = self._gethdf5arrays(outgroup)
            elif key == 'parameters' or key == 'Parameters':
                mydict = self._gethdf5attributes(hdfFile, key)
                self.cosmology = astropy.cosmology.LambdaCDM(H0   = mydict['H_0'],
                                                             Om0  = mydict['Omega_Matter'],
                                                             Ode0 = mydict['Omega_DE'])
                self.box_size=mydict['boxSize']   #already in Mpc
                self.sigma_8=mydict['sigma_8']
                self.n_s=mydict['N_s']

        # turam added - use first redshift
        #print self.catalog.items()[0], type(self.catalog.items()[0])
        #print self.catalog.values()[0]['redshift']
        self.redshift = self.catalog.values()[0]['redshift'][0]

        print "box_size after loading = ", self.box_size

        # TODO: how to get sky area?
        hdfFile.close()
        return self

    # Functions for applying filters

    def _check_halo(self, halo, filters):
        """
        Apply the requested filters to a given halo and return True if it
        passes them all, False if not.
        """
        status = True
        if type(filters) is not dict:
            raise TypeError("check_halo: filters must be given as dict")
        for filter_name in filters.keys():
            if filter_name == 'zlo':
                try:
                    zmax = max(halo['redshift'])
                    status = status and (zmax >= filters[filter_name])
                except KeyError:
                    status = False
            elif filter_name  == 'zhi':
                try:
                    zmin = min(halo['redshift'])
                    status = status and (zmin <= filters[filter_name])
                except KeyError:
                    status = False
        return status

    # Functions for returning quantities from the catalog

    def _get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        props = []
        for haloID in self.catalog.keys():
            halo = self.catalog[haloID]
            if self._check_halo(halo, filters):
                if quantity in halo.keys():
                    props.extend(halo[quantity])
        return np.asarray(props)

    def _get_derived_property(self, quantity, filters):
        """
        Return a derived halo property. These properties aren't stored
        in the catalog but can be computed from properties that are via
        a simple function call.
        """
        print "in get_derived_property: ", quantity, filters
        props = []

        #if 'position' in quantity:
        #  return np.asarray()

        stored_qty_rec = self.derived[quantity]
        stored_qty_name = stored_qty_rec[0]
        stored_qty_fctn = stored_qty_rec[1]
        print 'stored_qty:', stored_qty_name, stored_qty_fctn
        for haloID in self.catalog.keys():
            halo = self.catalog[haloID]
            print 'haloID = ', haloID, halo
            if self._check_halo(halo, filters):
                #if stored_qty_name in halo.keys():
                #    props.extend(stored_qty_fctn( halo[stored_qty_name] ))
                if type(stored_qty_name) is tuple and stored_qty_name[0] in halo.keys():
                    print 'branch1: ', quantity
                    values = halo[stored_qty_name[0]]
                    props.extend(stored_qty_fctn(values, stored_qty_name[1:]))
                else:
                    print 'branch2: ', quantity
                    if stored_qty_name in halo.keys():
                        props.extend(stored_qty_fctn( halo[stored_qty_name] ))
        return np.asarray(props)

    # Functions for computing derived values

    def _unlog10(self, propList):
        """
        Take a list of numbers and return 10.**(the numbers).
        """
        result = []
        for value in propList:
            result.append(10.**value)
        return result

    # HDF5 utility routines

    def _gethdf5keys(self,id,*args):
        keys=id.keys()
        keylist=[str(x) for x in keys]
        return keylist

    def _gethdf5attributes(self,id,key,*args):
        #Return dictionary with group attributes and values
        group=id[key]
        mydict={}
        for item in group.attrs.items():
            attribute=str(item[0])
            mydict[attribute]=item[1]
        #endfor
        return mydict

    def _gethdf5group(self,group,*args):
        #return dictionary of (sub)group dictionaries
        groupkeys=self._gethdf5keys(group)
        groupdict={}
        for key in groupkeys:
            mydict=self._gethdf5attributes(group,key)
            groupdict[str(key)]=mydict
        #endfor
        return groupkeys,groupdict

    def _gethdf5arrays(self,group,*args):
        groupkeys=self._gethdf5keys(group)
        arraydict={}
        oldlen=-1
        for key in groupkeys:
            array=np.array(group[key])
            arraylen=len(array)
            if(oldlen>-1):         #check that array length is unchanged
                if(oldlen!=arraylen):
                    print "Warning: hdf5 array length changed for key",key
                #endif
            else:
                oldlen=arraylen   #set to ist array length
            #endif
            arraydict[str(key)]=array
        #endfor
        return arraydict

    def _multiply(self, propList, factor_tuple):
        """
        Multiplication routine -- derived quantity is equal to a stored
        quantity times some factor. Additional args for the derived quantity
        routines are passed in as a tuple, so extract the factor first.
        """
        print "in _multiply: ", propList, " ; factor_tuple: ", factor_tuple
        factor = factor_tuple[0]
        return propList * factor

    def _add(self, propList):
        """
        Routine that returns element-wise addition of two arrays.
        """
        x = sum(propList)
        return x

