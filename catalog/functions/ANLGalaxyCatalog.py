# Argonne galaxy catalog class.

from GalaxyCatalogInterface import GalaxyCatalog
import numpy as np
import h5py
import astropy.cosmology

class ANLGalaxyCatalog(GalaxyCatalog):

    def __init__(self, fn=None):
        self.type_ext    = 'ANL'
        self.filters     = { 'zlo':                   True,
                             'zhi':                   True
                           }
        self.quantities  = { 'redshift':              self.get_stored_property,
                             'ra':                    self.get_stored_property,
                             'dec':                   self.get_stored_property,
                             'v_pec':                 self.get_stored_property,
                             'mass':                  self.get_stored_property,
                             'age':                   self.get_stored_property,
                             'stellar_mass':          self.get_derived_property,
                             'log_stellarmass':       self.get_stored_property,
                             'gas_mass':              self.get_stored_property,
                             'metallicity':           self.get_stored_property,
                             'sfr':                   self.get_stored_property,
                             'ellipticity':           self.get_stored_property,
                             'positionX':             self.get_stored_property,
                             'positionY':             self.get_stored_property,
                             'positionZ':             self.get_stored_property,
                             'velocityX':             self.get_stored_property,
                             'velocityY':             self.get_stored_property,
                             'velocityZ':             self.get_stored_property,
                             'disk_ra':               self.get_stored_property,
                             'disk_dec':              self.get_stored_property,
                             'disk_sigma0':           self.get_stored_property,
                             'disk_re':               self.get_stored_property,
                             'disk_index':            self.get_stored_property,
                             'disk_a':                self.get_stored_property,
                             'disk_b':                self.get_stored_property,
                             'disk_theta_los':        self.get_stored_property,
                             'disk_phi':              self.get_stored_property,
                             'disk_stellarmass':      self.get_derived_property,
                             'log_disk_stellarmass':  self.get_stored_property,
                             'disk_metallicity':      self.get_stored_property,
                             'disk_age':              self.get_stored_property,
                             'disk_sfr':              self.get_stored_property,
                             'disk_ellipticity':      self.get_stored_property,
                             'bulge_ra':              self.get_stored_property,
                             'bulge_dec':             self.get_stored_property,
                             'bulge_sigma0':          self.get_stored_property,
                             'bulge_re':              self.get_stored_property,
                             'bulge_index':           self.get_stored_property,
                             'bulge_a':               self.get_stored_property,
                             'bulge_b':               self.get_stored_property,
                             'bulge_theta_los':       self.get_stored_property,
                             'bulge_phi':             self.get_stored_property,
                             'bulge_stellarmass':     self.get_derived_property,
                             'log_bulge_stellarmass': self.get_stored_property,
                             'bulge_age':             self.get_stored_property,
                             'bulge_sfr':             self.get_stored_property,
                             'bulge_metallicity':     self.get_stored_property,
                             'bulge_ellipticity':     self.get_stored_property,
                             'agn_ra':                self.get_stored_property,
                             'agn_dec':               self.get_stored_property,
                             'agn_mass':              self.get_stored_property,
                             'agn_accretnrate':       self.get_stored_property,
                             'SDSS_u:rest:':          None,    # don't have a way to return these yet
                             'SDSS_g:rest:':          None,
                             'SDSS_r:rest:':          None,
                             'SDSS_i:rest:':          None,
                             'SDSS_z:rest:':          None,
                             'SDSS_u:observed:':      None,
                             'SDSS_g:observed:':      None,
                             'SDSS_r:observed:':      None,
                             'SDSS_i:observed:':      None,
                             'SDSS_z:observed:':      None,
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
        self.derived = {'stellar_mass':      ('log_stellarmass',       self.unlog10),
                        'disk_stellarmass':  ('log_disk_stellarmass',  self.unlog10),
                        'bulge_stellarmass': ('log_bulge_stellarmass', self.unlog10)}
        self.catalog = {}
        self.cosmology = None
        return GalaxyCatalog.__init__(self, fn)


    def load(self, fn):
        """
        Given a catalog path, attempt to read the catalog and set up its
        internal data structures.
        """

        hdfFile = h5py.File(fn, 'r')
        hdfKeys, hdfAttrs = self.gethdf5group(hdfFile)
        self.catalog = {}
        for key in hdfKeys:
            if 'Output' in key:
                outgroup = hdfFile[key]
                dataKeys, dataAttrs = self.gethdf5group(outgroup)
                self.catalog[key] = self.gethdf5arrays(outgroup)
            elif key == 'cosmology':
                mydict = self.gethdf5attributes(hdfFile, key)
                self.cosmology = astropy.cosmology.LambdaCDM(H0   = mydict['H_0'],
                                                             Om0  = mydict['Omega_Matter'],
                                                             Ode0 = mydict['Omega_DE'])
        hdfFile.close()
        return self

    # Functions for applying filters

    def check_halo(self, halo, filters):
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

    def get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the catalog as a NumPy
        array. This is for properties that are explicitly stored in the
        catalog.
        """
        props = []
        for haloID in self.catalog.keys():
            halo = self.catalog[haloID]
            if self.check_halo(halo, filters):
                if quantity in halo.keys():
                    props.extend(halo[quantity])
        return np.asarray(props)

    def get_derived_property(self, quantity, filters):
        """
        Return a derived halo property. These properties aren't stored
        in the catalog but can be computed from properties that are via
        a simple function call.
        """
        props = []
        stored_qty_rec = self.derived[quantity]
        stored_qty_name = stored_qty_rec[0]
        stored_qty_fctn = stored_qty_rec[1]
        for haloID in self.catalog.keys():
            halo = self.catalog[haloID]
            if self.check_halo(halo, filters):
                if stored_qty_name in halo.keys():
                    props.extend(stored_qty_fctn( halo[stored_qty_name] ))
        return np.asarray(props)

    # Functions for computing derived values

    def unlog10(self, propList):
        """
        Take a list of numbers and return 10.**(the numbers).
        """
        result = []
        for value in propList:
            result.append(10.**value)
        return result

    # HDF5 utility routines

    def gethdf5keys(self,id,*args):
        if(len(args)>0):
            blurb=args[0]
        else:
            blurb=None
        #endif                                                                                  
             
        keys=id.keys()
        keylist=[str(x) for x in keys]
        return keylist

    def gethdf5attributes(self,id,key,*args):
        #Return dictionary with group attributes and values                                     
             
        group=id[key]
        mydict={}
        for item in group.attrs.items():
            attribute=str(item[0])
            mydict[attribute]=item[1]

        #endfor                                                                                 
             
        return mydict

    def gethdf5group(self,group,*args):
        #return dictionary of (sub)group dictionaries                                           
             
        groupkeys=self.gethdf5keys(group)
        groupdict={}
        for key in groupkeys:
            mydict=self.gethdf5attributes(group,key)
            groupdict[str(key)]=mydict

        #endfor                                                                                 
             
        return groupkeys,groupdict

    def gethdf5arrays(self,group,*args):
        groupkeys=self.gethdf5keys(group)
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
