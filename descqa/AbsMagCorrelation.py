from __future__ import print_function, division, unicode_literals, absolute_import
import os
import math
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

import numpy as np
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt
from astropy.cosmology import WMAP9
import treecorr

__all__ = ['AngularCorrelation']




class AbsMagCorrelation(BaseValidationTest):
    """
    Validation test to show 2pt correlation function of ProtoDC2 with SDSS
    """
    def __init__(self, band='r', survey='sdss', observation='', 
                 RandomFactor=10, **kwargs):

        #catalog quantities
        self.mag_field = 'Mag_true_{}_{}_z0'.format(band, survey)
        self.band = band

        #validation data
        self.validation_data = {}
        self.observation = observation

        #Random number times shape of catalog
        self.RandomFactor = RandomFactor

        #setup summary plot
        self.first_pass = True

        self._other_kwargs = kwargs

        self.possible_observations = {
            'Zehavi2011_rAbsMagSDSS': {
                'filename_template': '2pt/Zehavi2011_SDSS_r_{}.dat',
                'usecols': (0, 1),
                'colnames': ('rp', 'wp'), #This is not the names of the columns in the data files but the name assigned to the columns 0 and 1
                'skiprows': 1,
                'label': 'Zehavi et al 2011',
            },
            'Wang2013_rAppMagSDSS': {
                'filename_template': '2pt/Wang2013_SDSS_fig15_r_{}.dat',
                'usecols': (0, 1),
                'colnames': ('theta', 'w'),
                'skiprows': 1,
                'label': 'Wang et al 2013',
            },
        }


    @staticmethod
    def get_catalog_data(gc, quantities, filters=None):
        '''
        Get catalog data
        '''
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True, summary='Missing requested quantities')

        data = gc.get_quantities(quantities, filters=filters)
        #make sure data entries are all finite
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data


    def get_validation_data(self, observation, maglimit):
        '''
        Get the observations
        '''
        data_args = self.possible_observations[observation]
        data_path = os.path.join(self.data_dir, data_args['filename_template'].format(maglimit))

        if not os.path.exists(data_path):
            raise ValueError("{}-maglimit data file {} not found".format(maglimit, data_path))

        if not os.path.getsize(data_path):
            raise ValueError("{}-maglimit data file {} is empty".format(maglimit, data_path))

        data = np.loadtxt(data_path, unpack=True, usecols=data_args['usecols'], skiprows=data_args['skiprows'])

        validation_data = dict(zip(data_args['colnames'], data))
        validation_data['label'] = data_args['label']

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        results = {}
        filenames = ['R_-20_-19.dat', 'R_-20_-21.dat', 'R_-21_-22.dat', 'R_-22_-23.dat']
        labels = [r'$-20<M_r<-19$', r'$-21<M_r<-20$', r'$-22<M_r<-21$', r'$-23<M_r<-22$']
        colors = plt.cm.jet(np.linspace(0, 1, len(filenames)))
        #Maximum magnitude, minimum luminosity
        Mag_r_maxs = [-19, -20, -21, -22]
        #Minimum magnitude, maximum luminosity
        Mag_r_mins = [-20, -21, -22, -23]
        #Redshift from Table 1 using cz_min and cz_max 
        zmins = [0.027, 0.042, 0.066, 0.103]
        zmaxs = [0.064, 0.106, 0.159, 0.245]
        min_sep = 1e-2 #Degree
        max_sep = 5 #Degree
        bin_size = 0.2
        h = 0.7
        for rmax, rmin, zmax, zmin, fname, label, color in zip(Mag_r_maxs, Mag_r_mins, zmaxs, zmins, filenames, labels, colors):
            #check for valid observations
            maglimit = '{}_{}'.format(rmin, rmax) 
            if not self.observation:
                print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
            elif self.observation not in self.possible_observations:
                raise ValueError('Observation {} not available'.format(self.observation))
            else:
                self.validation_data = self.get_validation_data(self.observation, maglimit)
            #Constrain in redshift and magnitude. DESQA accepts it as a function of corresponding quantities (in this case redshift and r-band magnitude)
            filters = [(lambda z: (z > zmin) & (z < zmax), 'redshift'), (lambda mag: (mag > rmin) & (mag < rmax), self.mag_field)] 
            #Generating catalog data based on the constrains and getting the columns ra and dec
            catalog_data = self.get_catalog_data(catalog_instance, ['ra', 'dec', 'redshift', self.mag_field], filters=filters)
            ra = catalog_data['ra']
            dec = catalog_data['dec']
            z = catalog_data['redshift']
            ramin, ramax = ra.min(), ra.max()
            decmin, decmax = dec.min(), dec.max()
            #z1, z2 = z.min(), z.max()

            #Giving ra and dec to treecorr 
            cat = treecorr.Catalog(ra=ra, dec=dec, r=h*WMAP9.comoving_distance(z).value, ra_units='degrees', dec_units='degrees')
            dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
            dd.process(cat, metric='Rperp')

            #Giving number of random points
            randomN = self.RandomFactor * ra.shape[0]

            rand_ra = np.random.uniform(low=ramin, high=ramax, size=randomN)
            rand_sindec = np.random.uniform(low=np.sin(decmin*np.pi/180.), high=np.sin(decmax*np.pi/180.), size=randomN)
            rand_dec = np.arcsin(rand_sindec) * 180. /np.pi
            rand_z = np.random.uniform(low=zmin, high=zmax, size=randomN)

            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, r=h*WMAP9.comoving_distance(rand_z).value, ra_units='degrees', dec_units='degrees')
            rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
            rr.process(rand_cat, metric='Rperp')

            #Random-Data 
            rd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
            rd.process(rand_cat, cat, metric='Rperp')

            #Data-Random
            dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
            dr.process(cat, rand_cat, metric='Rperp')

            dd.write(os.path.join(output_dir, fname), rr, dr, rd)
            xi, varxi = dd.calculateXi(rr, dr, rd)
            xi_rad = np.exp(dd.meanlogr)
            xi_sig = np.sqrt(varxi) 

            plt.errorbar(xi_rad, xi, xi_sig, marker='o', ls='', c=color) 
            plt.plot(self.validation_data['rp'], self.validation_data['wp'], c=color, label=label)
        plt.legend(loc=0)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$r_p (h^{-1}$ Mpc)')
        plt.ylabel(r'$w_p(r_p) (h^{-1}$ Mpc)')
        plt.text(1e-2, 50, 'Lines: Zehavi et al 2011')
        plt.text(1e-2, 10, 'Points: Catalog')
        plt.savefig(os.path.join(output_dir, 'proj_xi_magnitude.pdf'), bbox_inches='tight')

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        return TestResult(0, passed=True)


    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('meanz', keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <z>', keyname, keyname+'-', keyname+'+', ' '))
            else:
                fields = ('meanz', keyname)
                header = ', '.join(('Data columns are: <z>', keyname, ' '))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        pass
