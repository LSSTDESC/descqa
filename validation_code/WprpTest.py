from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as mp
from astropy import units as u

from ValidationTest import ValidationTest

class WprpTest(ValidationTest):
    """
    validaton test class object to compute project 2-point correlation function wp(rp)
    """
    
    def __init__(self, test_args, data_directory, data_args):
        """
        Initialize a stellar mass function validation test.
        
        Parameters
        ----------
        test_args : dictionary
            dictionary of arguments specifying the parameters of the test
            
            bins : tuple, optional
                minimum log mass, maximum log mass, N log bins
                default: (9.5,12.0,25)
            
            zlo : float, optional
                minimum redshift
                default: 0.0
            
            zhi : float, optional
                maximum redshift
                default: 1000.0
        
        data_directory : string
            path to comparison data directory
        
        data_args : dictionary
            dictionary of arguments specifying the comparison data
            
            file : string
            
            name : string
            
            usecols : tuple
            columns to use in data comparison file
            (bin centers, number_density, err)
        """
        
        super(ValidationTest, self).__init__()
        
        #set validation data information
        self._data_directory = data_directory
        self._data_file = data_args['file']
        self._data_name = data_args['name']
        self._data_args= data_args
        
        #load validation comparison data
        obinctr, ohist, ohmin, ohmax = self.load_validation_data()
        self.validation_data = (obinctr, ohist, ohmin, ohmax)
        
        #set parameters of test
        #stellar mass bins
        if 'bins' in list(test_args.keys()):
            min_m, max_m, N_bins = test_args['bins']
            self.mstar_bins = np.logspace(min_m, max_m, N_bins)
        else:
            self.mstar_bins = np.logspace(9.5, 12.0, 25)
        #minimum redshift
        if 'zlo' in list(test_args.keys()):
            zlo = test_args['zlo']
            self.zlo = float(zlo)
        else:
            self.zlo = 0.0
        #maximum redshift
        if 'zhi' in list(test_args.keys()):
            zhi = test_args['zhi']
            self.zhi = float(zhi)
        else:
            self.zhi = 1000.0
    
    def run_validation_test(self, galaxy_catalog, galaxy_catalog_name, output_dict):
        """
        Load galaxy catalog and (re)calculate the stellar mass function.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader
        
        galaxy_catalog_name : string
            name of mock galaxy catalog
        
        output_dict : dictionary
            dictionary of output informaton
        
        Returns
        -------
        test_passed : boolean
        """
        
        #make sure galaxy catalog has appropiate quantities
        if not 'stellar_mass' in galaxy_catalog.quantities.keys():
            #raise an informative warning
            msg = ('galaxy catalog does not have `stellar_mass` quantity, skipping the rest of the validation test.')
            warn(msg)
            #write to log file
            f = open(output_dict['log'], 'w')
            f.write(msg)
        else: #continue with the test
            
            #calculate stellar mass function in galaxy catalog
            binctr, binwid, mhist, mhmin, mhmax = self.binned_stellar_mass_function(galaxy_catalog)
            catalog_result = (binctr, binwid, mhist, mhmin, mhmax)
            
            #calculate summary statistic
            summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
            
            #plot results
            self.plot_result(catalog_result, galaxy_catalog_name, output_dict['figure'])
            
            #save results to files
            self.write_result_file(catalog_result, output_dict['catalog'])
            self.write_validation_file(self.validation_data, output_dict['validation'])
            self.write_summary_file(summary_result, output_dict['summary'])
            
            return test_passed
            
    def binned_stellar_mass_function(self, galaxy_catalog):
        """
        Calculate the stellar mass function.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get stellar masses from galaxy catalog
        masses = galaxy_catalog.get_quantities("stellar_mass", {'zlo': self.zlo, 'zhi': self.zhi})
        
        #remove non-finite r negative numbers
        mask = np.isfinite(masses) & (masses > 0.0)
        masses = masses[mask]
        
        #take log of masses
        logm = np.log10(masses)
        
        #count galaxies in log bins
        mhist, mbins = np.histogram(logm, bins=self.mstar_bins)
        Nbins = len(mbins)-1.0
        binctr = (mbins[1:] + mbins[:Nbins])/2.0
        binwid = mbins[1:] - mbins[:Nbins]
        
        #calculate volume
        if galaxy_catalog.lightcone:
            Vhi = galaxy_catalog.get_cosmology().comoving_volume(zhi)
            Vlo = galaxy_catalog.get_cosmology().comoving_volume(zlo)
            dV = float((Vhi - Vlo)/u.Mpc**3)
            # TODO: need to consider completeness in volume
            af = float(galaxy_catalog.get_sky_area() / (4.*np.pi*u.sr))
            vol = af * dV
        else:
            vol = galaxy_catalog.box_size**3.0
        
        #calculate number differential density
        mhmin = (mhist - np.sqrt(mhist)) / binwid / vol
        mhmax = (mhist + np.sqrt(mhist)) / binwid / vol
        mhist = mhist / binwid / vol
        
        return binctr, binwid, mhist, mhmin, mhmax
    
    def calulcate_summary_statistic(self, catalog_result):
        """
        Run summary statistic.
        
        Parameters
        ----------
        catalog_result :
        
        Returns
        -------
        result :  numerical result
        
        pass : boolean
            True if the test is passed, False otherwise.
        """
        
        return 1.0, True
    
    def load_validation_data(self):
        """
        Open comparsion validation data, i.e. observational comparison data.
        """
        
        fn = os.path.join(self._data_directory, self._data_file)
        
        binctr, mhist, merr = np.loadtxt(fn, unpack=True, usecols=self._data_args['usecols'])
        binctr = np.log10(binctr)
        mhmax = np.log10(mhist + merr)
        mhmin = np.log10(mhist - merr)
        mhist = np.log10(mhist)
        
        return binctr, mhist, mhmin, mhmax
    
    def plot_result(self, result, galaxy_catalog_name, savepath):
        """
        Create plot of stellar mass function
        
        Parameters
        ----------
        result :
            stellar mass function of galaxy catalog
        
        galaxy_catalog_name : string
            name of galaxy catalog
        
        savepath : string
            file to save plot
        """
        
        fig = mp.figure()
        
        #plot measurement from galaxy catalog
        sbinctr, sbinwid, shist, shmin, shmax = result
        mp.step(sbinctr, shist, where="mid", label=galaxy_catalog_name)
        
        #plot comparison data
        obinctr, ohist, ohmin, ohmax = self.validation_data
        mp.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist],
                    label=self._data_name, fmt='o')
        
        #add formatting
        mp.legend(loc='best', frameon=False)
        mp.title(r'stellar mass function')
        mp.xlabel(r'$\log M_*\ (M_\odot)$')
        mp.ylabel(r'$dN/dV\, d\log M\ ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$')
        
        #save plot
        fig.savefig(savepath)
    
    def write_result_file(self, result, savepath, comment=None):
        """
        write results to ascii files
        
        Parameters
        ----------
        result : 
        
        savepath : string
            file to save result
        
        comment : string
        """
        
        #unpack result
        binctr, binwid, hist, hmin, hmax = result
        
        #save result to file
        f = open(savepath, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h, hn, hx in zip(binctr, hist, hmin, hmax):
            f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
        f.close()
    
    def write_validation_file(self, result, savepath, comment=None):
        """
        write validation data to ascii files
        
        Parameters
        ----------
        result : 
        
        savepath : string
            file to save result
        
        comment : string
        """
        
        #unpack result
        binctr, hist, hmin, hmax = result
        
        #save result to file
        f = open(savepath, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h, hn, hx in zip(binctr, hist, hmin, hmax):
            f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
        f.close()


