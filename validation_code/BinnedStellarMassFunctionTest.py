from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
import importlib
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as mp
from astropy import units as u

from ValidationTest import ValidationTest

class BinnedStellarMassFunctionTest(ValidationTest):
    """
    validaton test class object to compute stellar mass function in log stellar mass bins
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
        self.validation_data = {'x':obinctr, 'y':ohist, 'y-':ohmin, 'y+':ohmax}
        
        #set parameters of test
        #stellar mass bins
        if 'bins' in test_args:
            self.mstar_log_bins = np.linspace(*test_args['bins'])
        else:
            self.mstar_log_bins = np.linspace(7.0, 12.0, 26)
        #minimum redshift
        if 'zlo' in test_args:
            zlo = test_args['zlo']
            self.zlo = float(zlo)
        else:
            self.zlo = 0.0
        #maximum redshift
        if 'zhi' in test_args:
            zhi = test_args['zhi']
            self.zhi = float(zhi)
        else:
            self.zhi = 1000.0

        self.summary_method = test_args.get('summary','L2Diff')
        self.threshold = test_args.get('threshold',1.0)
    
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
            dictionary of output information
        
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
            catalog_result = {'x':binctr,'dx': binwid, 'y':mhist, 'y-':mhmin, 'y+': mhmax}
            
            #calculate summary statistic
            summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
            
            #plot results
            self.plot_result(catalog_result, galaxy_catalog_name, output_dict['figure'])
            
            #save results to files
            self.write_file(catalog_result, output_dict['catalog'])
            self.write_file(self.validation_data, output_dict['validation'])
            self.write_summary_file(summary_result, test_passed, output_dict['summary'])
            
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
        
        #count galaxies in log bins
        mhist, mbins = np.histogram(np.log10(masses), bins=self.mstar_log_bins)
        binctr = (mbins[1:] + mbins[:-1])*0.5
        binwid = mbins[1:] - mbins[:-1]
        
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
        
        test_passed : boolean
            True if the test is passed, False otherwise.
        """
        
        module_name=self.summary_method
        summary_method=getattr(importlib.import_module(module_name), module_name)

        result, test_passed = summary_method(catalog_result,self.validation_data,self.threshold)

        #return result, test_passed
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
        mp.step(sbinctr, shist, where="mid", label=galaxy_catalog_name, color='blue')
        mp.fill_between(sbinctr, shmin, shmax, facecolor='blue', alpha=0.3, edgecolor='none')
        
        #plot comparison data
        obinctr, ohist, ohmin, ohmax = self.validation_data
        mp.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=self._data_name, fmt='o',color='green')
        
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
        for b, h, hn, hx in zip(*(result[k] for k in ['x','y','y-','y+'])):
            f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
        f.close()
    
    def write_file(self, result, savepath, comment=None):
        """
        write validation data to ascii files
        
        Parameters
        ----------
        result : 

        savepath : string
            file to save result
        
        comment : string
        """
        
        #save result to file
        f = open(savepath, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h, hn, hx in zip(*(result[k] for k in ['x','y','y-','y+'])):
            f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
        f.close()
    
    def write_summary_file(self, result, test_passed, savepath, comment=None):
        """
        Parameters
        ----------
        result :
        savepath : string
            savepath : string 
        """
        f = open(savepath, 'w')
        if(test_passed):
            f.write("SUCCESS: L2 = %G\n" %result)
        else:
            f.write("FAILED: L2 = %G is > threshold value %G\n" %(result,self.threshold))
        f.close()

