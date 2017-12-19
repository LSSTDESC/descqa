from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt
import treecorr
import healpy as hp
from astropy.cosmology import WMAP9


__all__ = ['CorrelationsTwoPoint']




class CorrelationsTwoPoint(BaseValidationTest):
    """
    Validation test to show 2pt correlation function of ProtoDC2 with SDSS
    """
    def __init__(self, band='r', observation='', RandomFactor=10, 
                 RemovePixels=200, possible_mag_fields='', **kwargs):

        #catalog quantities
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]
        self.band = band

        #Observation
        self.observation = observation

        #Random number times shape of catalog
        self.RandomFactor = RandomFactor
        self.RemovePixels = RemovePixels

        self.TestParams = kwargs

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

    @staticmethod
    def return_healpixel(ra, dec, nside, nest=False):
        '''
        Inputs: RA, DEC in degrees and nside of healpix 
        Return: array of pixel number
        '''
        theta = 90.0 - dec
        ra = ra * np.pi / 180.
        theta = theta * np.pi / 180.
        pixels = hp.ang2pix(nside, theta, ra, nest=nest)
        return pixels

    def modify_figure(self, xlabel, ylabel, output_dir, filename):
        plt.legend(loc=0)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(self.possible_observations[self.observation]['label'])
        plt.savefig(os.path.join(output_dir, '{:s}.png'.format(filename)), bbox_inches='tight')
        if self.TestParams['savepdf']:
            plt.savefig(os.path.join(output_dir, '{:s}.pdf'.format(filename)), bbox_inches='tight')


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        mag_field = catalog_instance.first_available(*self.possible_mag_fields)

        min_sep = self.TestParams['min_sep']
        max_sep = self.TestParams['max_sep']
        bin_size = self.TestParams['bin_size']
        nside = self.TestParams['nside']

        #Check whether the observations and magnitude systems are matching. 
        if not self.observation:
            raise ValueError('No observation requested')
        elif (self.observation == 'Zehavi2011_rAbsMagSDSS') & (mag_field.islower()):
            raise ValueError('Observation is in absolute magnitude given magnitude is apparent')
        elif (self.observation == 'Wang2013_rAppMagSDSS') & (mag_field.isupper()):
            raise ValueError('Observation is in apparent magnitude given magnitude is absolute')

        #Use Angular correlation function 
        if self.observation == 'Wang2013_rAppMagSDSS':
            filenames = self.TestParams['filenames']
            labels = self.TestParams['labels']
            colors = plt.cm.jet(np.linspace(0, 1, len(filenames)))
            mag_r_maxs = self.TestParams['mag_r_maxs']
            mag_r_mins = self.TestParams['mag_r_mins']

            #Filter on redshift space
            zfilter = (lambda z: (z < self.TestParams['redshift_max']), 'redshift')
            catalog_data_all = self.get_catalog_data(catalog_instance, ['ra', 'dec', mag_field], filters=[zfilter])
            if catalog_data_all is None:
                return TestResult(skipped=True, summary='Missing requested quantities')
            plt.figure()
            for rmax, rmin, fname, label, color in zip(mag_r_maxs, mag_r_mins, filenames, labels, colors):
                validation_data = self.get_validation_data(self.observation, '{}_{}'.format(rmin, rmax))
                catalog_data = GCRQuery((lambda mag: (mag > rmin) & (mag < rmax), mag_field)).filter(catalog_data_all)            
                ra = catalog_data['ra']
                dec = catalog_data['dec']

                #It works only with rectangle area of sky
                ramin, ramax = ra.min(), ra.max()
                decmin, decmax = dec.min(), dec.max()
                if ramin < 0:
                    ra = np.mod(ra, 360)

                del catalog_data

                #plt.scatter(ra, dec)
                #plt.savefig('s.png')

                #Giving ra and dec to treecorr 
                cat = treecorr.Catalog(ra=ra, dec=dec,  ra_units='degrees', dec_units='degrees')
                dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size, sep_units='degrees')
                dd.process(cat)

                #Giving number of random points
                randomN = self.RandomFactor * ra.shape[0]

                #Random points are made based on healpix. I find healpixels 
                #from original catalog. I remove healpixels with lower number
                #objects. I distribute the random points uniformly.

                #Original pixels
                original_pixels = self.return_healpixel(ra, dec, nside)
                original_pixels, original_pixels_count = np.unique(original_pixels, return_counts=True)
                #There is a uncertainty in the number 30 
                #original_pixels = original_pixels[original_pixels_count > original_pixels_count.min() * 30]
                #RemovePixelFactor = np.std(original_pixels_count)
                #original_pixels = original_pixels[original_pixels_count > original_pixels_count.min() + RemovePixelFactor]
                #There is a uncertainty in the number 200
                original_pixels = original_pixels[original_pixels_count > self.RemovePixels]
                #RA will have value 0 if RA>360. Therefore, to make the uniform
                #random values I add 360 to numbers less than 360. Then I find 
                #the minimum and maximum of RA and find uniform RA values.
                #After that I subtract 360 from values greater than 360 to make
                #sure that RA will have values from 0 to 360 
                if (ramin <=100) & (ramax >= 260):
                    ra = np.where(ra <= 100, ra+360, ra)
                    ramin = ra.min()
                    ramax = ra.max()
                    rand_ra = np.random.uniform(ramin, ramax, randomN)
                    rand_ra = np.where(rand_ra >= 360, rand_ra - 360, rand_ra)
                else:
                    rand_ra = np.random.uniform(low=ramin, high=ramax, size=randomN)
                rand_sindec = np.random.uniform(low=np.sin(decmin * np.pi / 180.), high=np.sin(decmax * np.pi / 180.), size=randomN)
                rand_dec = np.arcsin(rand_sindec) * 180. /np.pi
                rand_pixels = self.return_healpixel(rand_ra, rand_dec, nside)
                upixels = np.in1d(rand_pixels, original_pixels)
                rand_ra = rand_ra[upixels]
                rand_dec = rand_dec[upixels]

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
                plt.plot(validation_data['theta'], validation_data['w'], c=color, label=label)
            self.modify_figure(r'$\theta$ deg', r'$\xi(\theta)$', output_dir, 'xi_magnitude')

        if self.observation == 'Zehavi2011_rAbsMagSDSS':
            filenames = self.TestParams['filenames']
            labels = self.TestParams['labels']
            colors = plt.cm.jet(np.linspace(0, 1, len(filenames)))
            Mag_r_maxs = self.TestParams['Mag_r_maxs']
            Mag_r_mins = self.TestParams['Mag_r_mins']
            zmins = self.TestParams['zmins']
            zmaxs = self.TestParams['zmaxs']
            h = self.TestParams['h']

            plt.figure()
            for rmax, rmin, zmax, zmin, fname, label, color in zip(Mag_r_maxs, Mag_r_mins, zmaxs, zmins, filenames, labels, colors):
                #check for valid observations
                maglimit = '{}_{}'.format(rmin, rmax) 
                self.validation_data = self.get_validation_data(self.observation, maglimit)
                #Constrain in redshift and magnitude. DESQA accepts it as a function of corresponding quantities (in this case redshift and r-band magnitude)
                filters = [(lambda z: (z > zmin) & (z < zmax), 'redshift'), (lambda mag: (mag > rmin) & (mag < rmax), mag_field)] 
                #Generating catalog data based on the constrains and getting the columns ra and dec
                catalog_data = self.get_catalog_data(catalog_instance, ['ra', 'dec', 'redshift', mag_field], filters=filters)
                if catalog_data is None:
                    return TestResult(skipped=True, summary='Missing requested quantities')

                ra = catalog_data['ra']
                dec = catalog_data['dec']
                z = catalog_data['redshift']
                ramin, ramax = ra.min(), ra.max()
                decmin, decmax = dec.min(), dec.max()
                #z1, z2 = z.min(), z.max()
                if ramin < 0:
                    ra = np.mod(ra, 360)

                del catalog_data
                #Giving ra and dec to treecorr 
                cat = treecorr.Catalog(ra=ra, dec=dec, r=h*WMAP9.comoving_distance(z).value, ra_units='degrees', dec_units='degrees')
                dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, bin_size=bin_size)
                dd.process(cat, metric='Rperp')

                #Giving number of random points
                randomN = self.RandomFactor * ra.shape[0]

                #Original pixels
                original_pixels = self.return_healpixel(ra, dec, nside)
                original_pixels, original_pixels_count = np.unique(original_pixels, return_counts=True)
                original_pixels = original_pixels[original_pixels_count > self.RemovePixels]
                if (ramin <=100) & (ramax >= 260):
                    ra = np.where(ra <= 100, ra+360, ra)
                    ramin = ra.min()
                    ramax = ra.max()
                    rand_ra = np.random.uniform(ramin, ramax, randomN)
                    rand_ra = np.where(rand_ra >= 360, rand_ra - 360, rand_ra)
                else:
                    rand_ra = np.random.uniform(low=ramin, high=ramax, size=randomN)
                rand_sindec = np.random.uniform(low=np.sin(decmin*np.pi/180.), high=np.sin(decmax*np.pi/180.), size=randomN)
                rand_dec = np.arcsin(rand_sindec) * 180. /np.pi
                rand_pixels = self.return_healpixel(rand_ra, rand_dec, nside)
                upixels = np.in1d(rand_pixels, original_pixels)
                rand_ra = rand_ra[upixels]
                rand_dec = rand_dec[upixels]
                rand_z = np.random.uniform(low=zmin, high=zmax, size=randomN)
                rand_z = rand_z[upixels]

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
            self.modify_figure(r'$r_p (h^{-1}$ Mpc)', r'$w_p(r_p) (h^{-1}$ Mpc)', output_dir, 'proj_xi_magnitude')


        return TestResult(0, passed=True)

    def conclude_test(self, output_dir):
        pass

