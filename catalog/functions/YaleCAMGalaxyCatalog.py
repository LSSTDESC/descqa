# Yale CAM galaxy catalogue class
# Duncan Campbell
# Yale University
# February, 2016

# load modules
import os
import numpy as np
import h5py
from GalaxyCatalogInterface import GalaxyCatalog
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import re

__all__ = ['YaleCAMGalaxyCatalog']

class YaleCAMGalaxyCatalog(GalaxyCatalog):
    """
    Yale CAM galaxy catalog class.
    
    Notes
    -----
    The Yale CAM galaxy mocks store all physical properties internally in units where h=1.
    """
    
    def __init__(self, fn=None):
        """
        Initialize Yale CAM galaxy catalog class.
        
        Parameters
        ----------
        fn : string
            filename of mock catalog
        """
        
        # set file type and location
        self.type_ext = 'yale'
        self.root_path = '/global/project/projectdirs/lsst/descqa/'
        
        # set fixed properties
        self.lightcone = False
        self.cosmology = FlatLambdaCDM(H0=70.1, Om0 = 0.275)
        self.simulation = 'Massive Black'
        self.box_size = 100.0 / self.cosmology.h
        self.volume = self.box_size**3.0
        
        # translates between desc keywords to those used in the stored mock
        # note: all appropriate quantities are in h=1 units.
        self.quantities  = { 'stellar_mass': self._stored_property_wrapper('stellar_mass'),
                             'mass':         self._stored_property_wrapper('halo_mvir'),
                             'ssfr':         self._stored_property_wrapper('SSFR'),
                             'halo_id':      self._stored_property_wrapper('halo_id'),
                             'positionX':    self._stored_property_wrapper('x'),
                             'positionY':    self._stored_property_wrapper('y'),
                             'positionZ':    self._stored_property_wrapper('z'),
                             'velocityX':    self._stored_property_wrapper('vx'),
                             'velocityY':    self._stored_property_wrapper('vy'),
                             'velocityZ':    self._stored_property_wrapper('vz'),
                             'absmag_r':    self._stored_property_wrapper('absmag_r'),
                             'absmag_g':    self._stored_property_wrapper('absmag_g'),
                             'g-r':    self._stored_property_wrapper('g-r')
                           }
        
        return GalaxyCatalog.__init__(self, fn)
    
    def load(self, fn='yale_cam_age_matching_LiWhite_2009_z0.0.hdf5'):
        """
        load mock galaxy catalog
        
        Parameters
        ----------
        fn : string
            filename of mock catalog located at self.root_path
        """
        
        #extract mock parameters from filename
        nums = re.findall(r"[-+]?\d*\.\d+|\d+", fn)
        self.redshift = float(nums[-2])
        
        f = h5py.File(fn, 'r')
        toplevel=f.keys()[0]
        self._data = f.get(toplevel)
                        
        #convert quantities into physical units given the cosmology
        #see 'notes' section of the Yale CAM class.
        #see arXiv:1308.4150
        #cast hdf5 data as numpy array to allow modification of data without modifying file contents
        self.Xdata=np.array(self._data)
        self.Xdata['stellar_mass'] = self._data['stellar_mass']/(self.cosmology.h)**2
        self.Xdata['x'] = self._data['x']/(self.cosmology.h)
        self.Xdata['y'] = self._data['y']/(self.cosmology.h)
        self.Xdata['z'] = self._data['z']/(self.cosmology.h)
        self.Xdata['halo_mvir'] = self._data['halo_mvir']/(self.cosmology.h)
        self.Xdata['absmag_r'] = self._data['absmag_r'] + 5.0*np.log10(self.cosmology.h)
        self.Xdata['absmag_g'] = self._data['absmag_g'] + 5.0*np.log10(self.cosmology.h)
        
        #how many galaxies are in the catalog?
        self.Ngals = len(self._data)
        
        return self
    
    def _construct_mask(self, filters):
        """
        Construct a mask array for use in filtering the catalog.
        
        Parameters
        ----------
        filters: dict
            dictionary of filter constraints
        
        Returns
        -------
        mask : numpy.array
            numpy array boolean mask
        """
        
        #check that filters is of the correct type
        if type(filters) is not dict:
            msg = ('filters must be given as a dictionary type.')
            raise TypeError(msg)
        
        #initialize filter
        mask = np.ones((self.Ngals), dtype=bool)
        
        #generate boolean mask
        for filter_name in filters.keys():
            #place code here to create filter(s)
            pass
        
        return mask
    
    def _get_stored_property(self, quantity, filters):
        """
        Return the requested property of galaxies in the mock catalog.
        
        Parameters
        ----------
        quantity : string
            key into mock galaxy catalogue of galaxy property
        
        filters :  dict
            dictionary of filter constraints
        
        Returns
        -------
        property : numpy.array
            numpy array of requested property from the catalogue
        """
        
        #build filter
        filter_mask = self._construct_mask(filters)
        
        #return requested data as an array
        return self.Xdata[quantity][np.where(filter_mask)]
    
    def _stored_property_wrapper(self, name):
        """
        private function used to translate desc keywords into stored keywords in the mock
        
        Parameters
        ----------
        name : string
            key into stored mock catalogue
        
        """
        
        return (lambda quantity, filter : self._get_stored_property(name, filter))

