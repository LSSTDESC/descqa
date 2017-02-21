"""
"""

from __future__ import (division, print_function, absolute_import)
import numpy as np

from astropy import units as u

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
from warnings import warn

from ValidationTest import ValidationTest, TestResult
import CalcStats

__all__ = ['ValidationTestTemplate','plot_summary']
__author__ = []

catalog_output_file = 'catalog_test.txt'
validation_output_file = 'validation_test.txt'
summary_details_file = 'summary_details_test.txt'
summary_details_module = 'write_summary_details'
summary_output_file = 'summary_test.txt'
log_file = 'log_test.txt'
plot_file = 'plot_test.png'
plot_title = 'Stellar Mass Function'
xaxis_label = r'$\log (M^*/(M_\odot)$'
yaxis_label = r'$\log(dn/dV\,d\log M) ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$'
summary_colormap = 'rainbow'
test_range_color = 'red'

class ValidationTestTemplate(ValidationTest):
    """
    validaton test class object to compute stellar mass function bins
    """
    
    def __init__(self, **kwargs):
        """
        initialize a validation test
        
        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots
        
        test_name : string
            string indicating test name
        
        observation : string, optional
            name of validation observation:
        
        bins : tuple, optional
        
        zlo : float, optional
        
        zhi : float, optional

        summary_details : boolean, optional

        summary_method : string, optional

        threshold : float, optional

        validation_range : tuple, optional
        """
        
        super(self.__class__, self).__init__(**kwargs)
        
        #load validation data
        if 'observation' in kwargs:
            available_observations = ['TestObservation']
            if kwargs['observation'] in available_observations:
                self.observation = kwargs['observation']
            else:
                msg = ('`observation` not available')
                raise ValueError(msg)
        else:
            self.observation = 'TestObservation'
        
        obinctr, ohist, ohmin, ohmax = self.load_validation_data()
        #bin center, number density, lower bound, upper bound
        self.validation_data = {'x':obinctr, 'y':ohist, 'y-':ohmin, 'y+':ohmax}
        
        #setup binning for observable
        if 'bins' in kwargs:
            self.obs_log_bins = np.linspace(*kwargs['bins'])
        else:
            self.obs_log_bins = np.linspace(7.0, 12.0, 26)  #default binning
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

        #set remaining parameters
        self.summary_method = kwargs.get('summary','L2Diff')  #test statistic
        self.threshold = kwargs.get('threshold',1.0)          #pass-fail threshold
        self.summary_details = kwargs.get('summary_details',False)  #detailed output
        self.validation_range = kwargs.get('validation_range',(7.0,12.0)) #test range
        
    def load_validation_data(self):
        """
        load tabulated stellar mass function data
        """
        
        #associate files with observations
        validation_data_files = {'file1':'path_to_data_directory/datafile_1.txt',
                                      'file2':'path_to_data_directory/datafile_2.txt'}
        
        #set the columns to use in each file
        columns = {'file1':(0,5,6),
                   'file2':(0,1,2),}
        
        #get path to file
        fn = os.path.join(self.base_data_dir, validation_data_files[self.observation])
        
        #column 1: bin center
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
        if not 'test_observable' in galaxy_catalog.quantities:
            #raise an informative warning
            msg = ('galaxy catalog does not have `test_observable` quantity, skipping the rest of the validation test.')
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir ,log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return TestResult('SKIPPED', msg)
        
        #calculate test observable in galaxy catalog
        binctr, binwid, mhist, mhmin, mhmax = self.binned_observable(galaxy_catalog)
        catalog_result = {'x':binctr,'dx': binwid, 'y':mhist, 'y-':mhmin, 'y+': mhmax}
        
        #calculate summary statistic and write detailes summary file if requested
        summary_result, test_passed, test_details = self.calculate_summary_statistic(catalog_result,details=self.summary_details)
        if (self.summary_details):
            fn = os.path.join(base_output_dir, summary_details_file)
            write_summary_details=getattr(CalcStats, summary_details_module)
            write_summary_details(test_details, fn, method=self.summary_method, comment='')
        
        #plot results
        fn = os.path.join(base_output_dir ,plot_file)
        self.plot_result(catalog_result, catalog_name, fn, test_details=test_details)
        
        #save results to files
        fn = os.path.join(base_output_dir, catalog_output_file)
        self.write_file(catalog_result, fn)
        
        fn = os.path.join(base_output_dir, validation_output_file)
        self.write_file(self.validation_data, fn)
        
        fn = os.path.join(base_output_dir, summary_output_file)
        self.write_summary_file(summary_result, test_passed, fn)

        msg = "{} = {:G} {} {:G}".format(self.summary_method, summary_result, '<' if test_passed else '>', self.threshold)
        return TestResult(summary_result,msg,test_passed)
    
    def binned_observable(self, galaxy_catalog):
        """
        calculate observable in bins
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get stellar catalog_data from galaxy catalog
        catalog_data = galaxy_catalog.get_quantities("test_observable", {'zlo': self.zlo, 'zhi': self.zhi})
        
        #remove non-finite r negative numbers
        mask = np.isfinite(catalog_data) & (catalog_data > 0.0)
        catalog_data = catalog_data[mask]
        
        #count galaxies in log bins
        mhist, mbins = np.histogram(np.log10(catalog_data), bins=self.obs_log_bins)
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
    
    
    def calculate_summary_statistic(self, catalog_result, details=False):
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
        summary_method=getattr(CalcStats, module_name)
        
       #restrict range of validation data supplied for test if necessary
        mask = (self.validation_data['x']>self.validation_range[0]) & (self.validation_data['x']<self.validation_range[1])
        if all(mask):
            validation_data = self.validation_data
        else:
            validation_data={}
            for k in self.validation_data:
                validation_data[k] = self.validation_data[k][mask]

        test_details={}
        if details:
            result, test_passed, test_details = summary_method(catalog_result, validation_data, self.threshold, details=details)
        else:
            result, test_passed = summary_method(catalog_result, validation_data, self.threshold)
        
        return result, test_passed, test_details
    

    def plot_result(self, result, catalog_name, savepath, test_details={}):
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
        line1, = plt.step(sbinctr, shist, where="mid", label=catalog_name, color='blue')
        plt.fill_between(sbinctr, shmin, shmax, facecolor='blue', alpha=0.3, edgecolor='none')
        
        #plot comparison data
        obinctr, ohist, ohmin, ohmax = (self.validation_data['x'], self.validation_data['y'], self.validation_data['y-'], self.validation_data['y+'])
        pts1 = plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=self.observation, fmt='o',color='green')
        
        #add validation region to plot
        if len(test_details)>0:          #xrange from test_details
            xrange=test_details['x']
        else:                            #xrange from validation_range
            mask = (self.validation_data['x']>self.validation_range[0]) & (self.validation_data['x']<self.validation_range[1])
            xrange = self.validation_data['x'][mask]
        ymin,ymax=plt.gca().get_ylim()
        plt.fill_between(xrange, ymin, ymax, color=test_range_color, alpha=0.15)
        patch=mpatches.Patch(color=test_range_color, alpha=0.1, label='Test Region') #create color patch for legend
        handles=[line1,pts1,patch]

        #add formatting
        plt.legend(handles=handles, loc='best', frameon=False, numpoints=1)
        plt.title(plot_title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)
        
        #save plot
        fig.savefig(savepath)
        plt.close(fig)
    
    
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


def plot_summary(output_file, catalog_list, validation_kwargs):
    """
    make summary plot for validation test

    Parameters
    ----------
    output_file: string
        filename for summary plot
    
    catalog_list: list of tuple
        list of (catalog, catalog_output_dir) used for each catalog comparison
    
    validation_kwargs : dict
        keyword arguments used in the validation
    """

    #initialize plot
    fig = plt.figure()
    plt.title(plot_title)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)    

    #setup colors from colormap
    colors= matplotlib.cm.get_cmap('nipy_spectral')(np.linspace(0.,1.,len(catalog_list)))
    
    #loop over catalogs and plot
    for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
        fn = os.path.join(catalog_dir, catalog_output_file)
        sbinctr, shist, shmin, shmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
        plt.step(sbinctr, shist, where="mid", label=catalog_name, color=color)
        plt.fill_between(sbinctr, shmin, shmax, facecolor=color, alpha=0.3, edgecolor='none')
    
    #plot 1 instance of validation data (same for each catalog)
    fn = os.path.join(catalog_dir, validation_output_file)
    obinctr, ohist,ohmin, ohmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
    plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=validation_kwargs['observation'], fmt='o',color='black')
    plt.legend(loc='best', frameon=False,  numpoints=1, fontsize='small')
    
    plt.savefig(output_file)
    plt.close(fig)