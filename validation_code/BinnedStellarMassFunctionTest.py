"""
"""

from __future__ import (division, print_function, absolute_import)
import numpy as np

from astropy import units as u

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt

import os
import importlib
from warnings import warn

from ValidationTest import ValidationTest, TestResult

__all__ = ['BinnedStellarMassFunctionTest','plot_summary']
__author__ = []

catalog_output_file = 'catalog_smf.txt'
validation_output_file = 'validation_smf.txt'
summary_output_file = 'summary_smf.txt'
log_file = 'log_smf.txt'
plot_file = 'plot_smf.png'

class BinnedStellarMassFunctionTest(ValidationTest):
    """
    validaton test class object to compute stellar mass function bins
    """
    
    def __init__(self, **kwargs):
        """
        initialize a stellar mass function validation test
        
        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots
        
        test_name : string
            string indicating test name
        
        observation : string, optional
            name of stellar mass validation observation:
            LiWhite2009, MassiveBlackII
        
        bins : tuple, optional
        
        zlo : float, optional
        
        zhi : float, optional
        """
        
        super(self.__class__, self).__init__(**kwargs)
        
        #load validation data
        if 'observation' in kwargs:
            available_observations = ['LiWhite2009', 'MassiveBlackII']
            if kwargs['observation'] in available_observations:
                self.observation = kwargs['observation']
            else:
                msg = ('`observation` not available')
                raise ValueError(msg)
        else:
            self.observation = 'LiWhite2009'
        
        obinctr, ohist, ohmin, ohmax = self.load_validation_data()
        #bin center, number density, lower bound, upper bound
        self.validation_data = {'x':obinctr, 'y':ohist, 'y-':ohmin, 'y+':ohmax}
        
        #stellar mass bins
        if 'bins' in kwargs:
            self.mstar_log_bins = np.linspace(*kwargs['bins'])
        else:
            self.mstar_log_bins = np.linspace(7.0, 12.0, 26)
        #minimum redshift
        if 'zlo' in kwargs:
            zlo = kwargs['zlo']
            self.zlo = float(zlo)
        else:
            self.zlo = 0.0
        #maximum redshift
        if 'zhi' in kwargs:
            zhi = kwargs['zhi']
            self.zhi = float(zhi)
        else:
            self.zhi = 1000.0
        
        self.summary_method = kwargs.get('summary','L2Diff')
        self.threshold = kwargs.get('threshold',1.0)
        
        
    def load_validation_data(self):
        """
        load tabulated stellar mass function data
        """
        
        #associate files with observations
        stellar_mass_fuction_files = {'LiWhite2009':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt',
                                      'MassiveBlackII':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt'}
        
        #set the columns to use in each file
        columns = {'LiWhite2009':(0,5,6),
                   'MassiveBlackII':(0,1,2),}
        
        #get path to file
        fn = os.path.join(self.base_data_dir, stellar_mass_fuction_files[self.observation])
        
        #column 1: stellar mass bin center
        #column 2: number density
        #column 3: 1-sigma error
        binctr, mhist, merr = np.loadtxt(fn, unpack=True, usecols=columns[self.observation])
        
        #take log of values
        binctr = np.log10(binctr)
        mhmax = np.log10(mhist + merr)
        mhmin = np.log10(mhist - merr)
        mhist = np.log10(mhist)
        
        return binctr, mhist, mhmin, mhmax
    
    
    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        run the validation test
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader
        
        catalog_name : string
            name of galaxy catalog
        
        Returns
        -------
        test_result : TestResult object
            use the TestResult object to reture test result
        """
        
        #make sure galaxy catalog has appropiate quantities
        if not 'stellar_mass' in galaxy_catalog.quantities:
            #raise an informative warning
            msg = ('galaxy catalog does not have `stellar_mass` quantity, skipping the rest of the validation test.')
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir ,log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return TestResult('SKIPPED', msg)
        
        #calculate stellar mass function in galaxy catalog
        binctr, binwid, mhist, mhmin, mhmax = self.binned_stellar_mass_function(galaxy_catalog)
        catalog_result = {'x':binctr,'dx': binwid, 'y':mhist, 'y-':mhmin, 'y+': mhmax}
        
        #calculate summary statistic
        summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
        
        #plot results
        fn = os.path.join(base_output_dir ,plot_file)
        self.plot_result(catalog_result, catalog_name, fn)
        
        #save results to files
        fn = os.path.join(base_output_dir, catalog_output_file)
        self.write_file(catalog_result, fn)
        
        fn = os.path.join(base_output_dir, validation_output_file)
        self.write_file(self.validation_data, fn)
        
        fn = os.path.join(base_output_dir, summary_output_file)
        self.write_summary_file(summary_result, test_passed, fn)

        msg = "{} = {:G} {} {:G}".format(self.summary_method, summary_result, '<' if test_passed else '>', self.threshold)
        return TestResult('PASSED' if test_passed else 'FAILED', msg)
    
    def binned_stellar_mass_function(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins
        
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
        mhist = np.log10(mhist)
        mhmin = np.log10(mhmin)
        mhmax = np.log10(mhmax)
        
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
        
        return result, test_passed
    
    
    def plot_result(self, result, catalog_name, savepath):
        """
        plot the stellar mass function of the catalog and validation data
        
        Parameters
        ----------
        result : dictionary
            stellar mass function of galaxy catalog
        
        catalog_name : string
            name of galaxy catalog
        
        savepath : string
            file to save plot
        """
        
        fig = plt.figure()
        
        #plot measurement from galaxy catalog
        sbinctr, sbinwid, shist, shmin, shmax = (result['x'], result['dx'], result['y'], result['y-'], result['y+'])
        plt.step(sbinctr, shist, where="mid", label=catalog_name, color='blue')
        plt.fill_between(sbinctr, shmin, shmax, facecolor='blue', alpha=0.3, edgecolor='none')
        
        #plot comparison data
        obinctr, ohist, ohmin, ohmax = (self.validation_data['x'], self.validation_data['y'], self.validation_data['y-'], self.validation_data['y+'])
        plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=self.observation, fmt='o',color='green')
        
        #add formatting
        plt.legend(loc='best', frameon=False)
        plt.title(r'stellar mass function')
        plt.xlabel(r'$\log M_*\ (M_\odot)$')
        plt.ylabel(r'$dN/dV\, d\log M\ ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$')
        
        #save plot
        fig.savefig(savepath)
    
    
    def write_file(self, result, filename, comment=None):
        """
        write validation steller mass function data file
        
        Parameters
        ----------
        result : dictionary
        
        filename : string
        
        comment : string
        """
        
        #save result to file
        f = open(filename, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h, hn, hx in zip(*(result[k] for k in ['x','y','y-','y+'])):
            f.write("%13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx))
        f.close()
    
    
    def write_summary_file(self, result, test_passed, filename, comment=None):
        """
        write summary data file
        
        Parameters
        ----------
        result : float
        
        test_passed : boolean
        
        filename : string
        
        comment : string, optional
        """
        
        #save result to file
        f = open(filename, 'w')
        if(test_passed):
            f.write("SUCCESS: %s = %G\n" %(self.summary_method, result))
        else:
            f.write("FAILED: %s = %G is > threshold value %G\n" %(self.summary_method, result, self.threshold))
        f.close()


def plot_summary(output_file, test_dicts):
    """
    make summary plot for validation test
    
    Parameters
    ----------
    output_file: string
        filename for summary plot
    
    test_dict: list
        list of dictionaries use to run each catalog comparison
    """
    
    #initialize plot
    fig = plt.figure()
    plt.title(r'stellar mass function')
    plt.xlabel(r'$\log M_*\ (M_\odot)$')
    plt.ylabel(r'$dN/dV\, d\log M\ ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$')
    
    #setup colors from colormap
    ncatalogs= len(test_dicts)
    cmap = matplotlib.cm.get_cmap('nipy_spectral')
    colors= cmap(np.linspace(0.,1.,ncatalogs))
    
    #loop over catalogs and plot
    for td,tdict in enumerate(test_dicts):
        fn = os.path.join(tdict['base_output_dir'],catalog_output_file)
        galaxy_catalog_name=tdict['catalog_name']
        sbinctr, shist, shmin, shmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
        plt.step(sbinctr, shist, where="mid", label=galaxy_catalog_name, color=colors[td])
        plt.fill_between(sbinctr, shmin, shmax, facecolor=colors[td], alpha=0.3, edgecolor='none')
    
    #plot 1 instance of validation data (same for each catalog)
    fn = os.path.join(test_dicts[0]['base_output_dir'],validation_output_file)
    obinctr, ohist,ohmin, ohmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
    plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=test_dicts[0]['observation'], fmt='o',color='black')
    plt.legend(loc='best', frameon=False)
    
    plt.savefig(output_file)
    
    return
