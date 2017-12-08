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
import treecorr

__all__ = ['AngularCorrelation']




class AngularCorrelation(BaseValidationTest):
    """
    Validation test to show 2pt correlation function of ProtoDC2 with SDSS
    """
    def __init__(self, band='r', observation='', RandomFactor=10, **kwargs):

        #catalog quantities
        possible_mag_fields = ('mag_{}_sdss',
                               'mag_{}_des',
                               'mag_{}_lsst',
                              )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]

        self.band = band
        self.redshift_max = 0.3

        #validation data
        self.validation_data = {}
        self.observation = observation

        #Random number times shape of catalog
        self.RandomFactor = RandomFactor

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
            return None

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
        mag_field = catalog_instance.first_available(*self.possible_mag_fields)
        filenames = ['r17_18.dat', 'r18_19.dat', 'r19_20.dat', 'r20_21.dat', 'r17_21.dat']
        labels = [r'$r=17-18$', r'$r=18-19$', r'$r=19-20$', r'$r=20-21$', r'$r=17-21$']
        colors = plt.cm.jet(np.linspace(0, 1, len(filenames)))
        #Maximum magnitude, minimum luminosity
        mag_r_maxs = [18, 19, 20, 21, 21]
        #Minimum magnitude, maximum luminosity
        mag_r_mins = [17, 18, 19, 20, 17]
        min_sep = 1e-2 #Degree
        max_sep = 5 #Degree
        bin_size = 0.2
        #Filter on redshift space
        zfilter = (lambda z: (z < self.redshift_max), 'redshift')
        for rmax, rmin, fname, label, color in zip(mag_r_maxs, mag_r_mins, filenames, labels, colors):
            #check for valid observations
            maglimit = '{}_{}'.format(rmin, rmax) 
            if not self.observation:
                print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
            elif self.observation not in self.possible_observations:
                raise ValueError('Observation {} not available'.format(self.observation))
            else:
                self.validation_data = self.get_validation_data(self.observation, maglimit)
            #Constrain in redshift and magnitude. DESQA accepts it as a function of corresponding quantities (in this case redshift and r-band magnitude)
            filters = [zfilter, (lambda mag: (mag > rmin) & (mag < rmax), mag_field)] 
            #Generating catalog data based on the constrains and getting the columns ra and dec
            catalog_data = self.get_catalog_data(catalog_instance, ['ra', 'dec'], filters=filters)
            if catalog_data is None:
                return TestResult(skipped=True, summary='Missing requested quantities')
            ra = catalog_data['ra']
            dec = catalog_data['dec']

            ramin, ramax = ra.min(), ra.max()
            decmin, decmax = dec.min(), dec.max()

            #Giving ra and dec to treecorr 
            cat = treecorr.Catalog(ra=ra, dec=dec,  ra_units='degrees', dec_units='degrees')
            dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
            dd.process(cat)

            #Giving number of random points
            randomN = self.RandomFactor * ra.shape[0]

            rand_ra = np.random.uniform(low=ramin, high=ramax, size=randomN)
            rand_sindec = np.random.uniform(low=np.sin(decmin*np.pi/180.), high=np.sin(decmax*np.pi/180.), size=randomN)
            rand_dec = np.arcsin(rand_sindec) * 180. /np.pi

            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='degrees', dec_units='degrees')
            rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
            rr.process(rand_cat)

            #Random-Data 
            rd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
            rd.process(rand_cat, cat)

            #Data-Random
            dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
            dr.process(cat, rand_cat)

            dd.write(os.path.join(output_dir, fname), rr, dr, rd)
            xi, varxi = dd.calculateXi(rr, dr, rd)
            xi_rad = np.exp(dd.meanlogr)
            xi_sig = np.sqrt(varxi) 

            plt.errorbar(xi_rad, xi, xi_sig, marker='o', ls='', c=color) 
            plt.plot(self.validation_data['theta'], self.validation_data['w'], c=color, label=label)
        plt.legend(loc=0)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\theta$ deg')
        plt.ylabel(r'$\xi(\theta)$')
        plt.text(1e-2, 5e-3, 'Lines: Wang et al 2013')
        plt.text(1e-2, 2e-3, 'Points: Catalog')
        plt.savefig(os.path.join(output_dir, 'xi_magnitude.pdf'), bbox_inches='tight')

        return TestResult(0, passed=True)

    def conclude_test(self, output_dir):
        pass