from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
from astropy import units as u

from ValidationTest import ValidationTest, TestResult

catalog_output_file = 'catalog.txt'
validation_output_file = 'validation.txt'
summary_output_file = 'summary.txt'
log_file = 'log.txt'
plot_file = 'plot.png'

class ColorDistributionTest(ValidationTest):
    """
    validaton test class object to compute galaxy color distribution
    """
    
    def __init__(self, **kwargs):
        """
        Initialize a color distribution validation test.
        
        Parameters
        ----------

        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots

        bins : tuple, required
            minimum color, maximum color, N bins

        limiting_band: string, optional
            band of the magnitude limit in the validation catalog

        limiting_mag: float, optional
            the magnitude limit

        zlo : float, requred
            minimum redshift of the validation catalog
        
        zhi : float, requred
            maximum redshift of the validation catalog

        band1 : string, required
            the first photometric band

        band2 : string, required
            the second photometric band
                            
        datafile : string
            path to the validation data file
            
        dataname : string
            name of the validation data
            
        """
        
        super(ValidationTest, self).__init__()
        
        #set validation data information
        self._data_file = os.path.join(kwargs['base_data_dir'], kwargs['datafile'])
        self._data_name = kwargs['dataname']
        
        #load validation comparison data
        binctr, hist = self.load_validation_data()
        self.validation_data = (binctr, hist)
        
        #set parameters of test
        #color bins
        if 'bins' in kwargs:
            self.color_bins = np.linspace(*kwargs['bins'])
        else:
            raise ValueError('bins not found!')
        #band of limiting magnitude
        if 'limiting_band' in list(kwargs.keys()):
            self.limiting_band = kwargs['limiting_band']
        else:
            self.limiting_band = None
        #limiting magnitude
        if 'limiting_mag' in list(kwargs.keys()):
            self.limiting_mag = kwargs['limiting_mag']
        else:
            self.limiting_mag = None
        #minimum redshift
        if 'zlo' in list(kwargs.keys()):
            zlo = kwargs['zlo']
            self.zlo = float(zlo)
        else:
            raise ValueError('zlo not found!')
        #maximum redshift
        if 'zhi' in list(kwargs.keys()):
            zhi = kwargs['zhi']
            self.zhi = float(zhi)
        else:
            raise ValueError('zhi not found!')

        #the first photometric band
        if 'band1' in list(kwargs.keys()):
            band1 = kwargs['band1']
            self.band1 = band1
        else:
            raise ValueError('band1 not found!')
        #the second photometric band
        if 'band2' in list(kwargs.keys()):
            band2 = kwargs['band2']
            self.band2 = band2
        else:
            raise ValueError('band2 not found!')

    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        run the validation test
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader
        
        catalog_name : string
            name of mock galaxy catalog
        
        Returns
        -------
        test_result : TestResult obj
        """
        
        #make sure galaxy catalog has appropiate quantities
        if not all(k in galaxy_catalog.quantities for k in (self.band1, self.band2)):
            #raise an informative warning
            msg = ('galaxy catalog does not have `band1` quantity, skipping the rest of the validation test.')
            warn(msg)
            return TestResult('SKIPPED', msg)

        #calculate color distribution in galaxy catalog
        binctr, hist = self.color_distribution(galaxy_catalog)
        if binctr is None:
            return TestResult('SKIPPED', 'nothing in the catalog')
        catalog_result = (binctr, hist)
        
        #calculate summary statistic
        summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
        
        #plot results
        fn = os.path.join(base_output_dir, plot_file)
        self.plot_result(catalog_result, catalog_name, fn)
        
        #save results to files
        fn = os.path.join(base_output_dir, catalog_output_file)
        self.write_file(catalog_result, fn)

        fn = os.path.join(base_output_dir, validation_output_file)
        self.write_file(self.validation_data, fn)

        fn = os.path.join(base_output_dir, summary_output_file)
        self.write_summary_file(summary_result, fn)
            
        return TestResult('PASSED', 'summary statistics not yet implemented')
            
    def color_distribution(self, galaxy_catalog):
        """
        Calculate the color distribution.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get magnitudes from galaxy catalog
        mag1 = galaxy_catalog.get_quantities(self.band1, {'zlo': self.zlo, 'zhi': self.zhi})
        mag2 = galaxy_catalog.get_quantities(self.band2, {'zlo': self.zlo, 'zhi': self.zhi})
        

        #apply magnitude limit
        if self.limiting_band is not None:
            mag_lim = galaxy_catalog.get_quantities(self.limiting_band, {'zlo': self.zlo, 'zhi': self.zhi})
            mask = mag_lim<self.limiting_mag
            mag1 = mag1[mask]
            mag2 = mag2[mask]


        if np.sum(mask)==0:
            msg = 'No object in the magnitude range!'
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return None, None

        #remove nonsensical magnitude values
        mask = (mag1>0) & (mag1<50) & (mag2>0) & (mag2<50)
        mag1 = mag1[mask]
        mag2 = mag2[mask]

        if np.sum(mask)==0:
            warn('No object in the redshift range!')
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return None, None
                    
        #count galaxies
        hist, bins = np.histogram(mag1-mag2, bins=self.color_bins)
        #normalize the histogram so that the sum of hist is 1
        hist = hist/np.sum(hist)
        Nbins = len(bins)-1.0
        binctr = (bins[1:] + bins[:Nbins])/2.0
        
        return binctr, hist
    
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
        binctr, hist = np.loadtxt(self._data_file, unpack=True)
        
        return binctr, hist
    
    def plot_result(self, result, catalog_name, savepath):
        """
        Create plot of color distribution
        
        Parameters
        ----------
        result :
            plot of color distribution of mock catalog and observed catalog
        
        catalog_name : string
            name of galaxy catalog
        
        savepath : string
            file to save plot
        """
        
        fig = plt.figure()
        
        #plot measurement from galaxy catalog
        mbinctr, mhist = result
        plt.step(mbinctr, mhist, where="mid", label=catalog_name, color='blue')
        
        #plot comparison data
        obinctr, ohist = self.validation_data
        plt.step(obinctr, ohist, label=self._data_name,color='green')
        
        #add formatting
        plt.legend(loc='best', frameon=False)
        plt.title('color distribution')
        plt.xlabel(self.band1+'-'+self.band2)
        # plt.ylabel(r'')
        
        #save plot
        fig.savefig(savepath)
    
    def write_file(self, result, savepath, comment=None):
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
        binctr, hist = result
        
        #save result to file
        f = open(savepath, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h in zip(binctr, hist):
            f.write("%13.6e %13.6e\n" % (b, h))
        f.close()
    
    def write_summary_file(self, result, savepath, comment=None):
        """
        
        """
        pass
        

