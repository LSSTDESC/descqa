from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
from astropy import units as u

from ValidationTest import ValidationTest

class ColorDistributionTest(ValidationTest):
    """
    validaton test class object to compute galaxy color distribution
    """
    
    def __init__(self, test_args, data_directory, data_args):
        """
        Initialize a color distribution validation test.
        
        Parameters
        ----------
        test_args : dictionary
            dictionary of arguments specifying the parameters of the test
                        
            bins : tuple, required
                minimum color, maximum color, N bins

            zlo : float, requred
                minimum redshift
            
            zhi : float, requred
                maximum redshift

            band1: string, required
                the first photometric band

            band2: string, required
                the second photometric band
        
        data_directory : string
            path to comparison data directory
        
        data_args : dictionary
            dictionary of arguments specifying the comparison data
            
            file : string
            
            name : string
            
        """
        
        super(ValidationTest, self).__init__()
        
        #set validation data information
        self._data_directory = data_directory
        self._data_file = data_args['file']
        self._data_name = data_args['name']
        self._data_args= data_args
        
        #load validation comparison data
        binctr, hist = self.load_validation_data()
        self.validation_data = (binctr, hist)
        
        #set parameters of test
        #color bins
        if 'bins' in test_args:
            self.color_bins = np.linspace(*test_args['bins'])
        else:
            raise ValueError('bins not found!')
        #minimum redshift
        if 'zlo' in list(test_args.keys()):
            zlo = test_args['zlo']
            self.zlo = float(zlo)
        else:
            raise ValueError('zlo not found!')
        #maximum redshift
        if 'zhi' in list(test_args.keys()):
            zhi = test_args['zhi']
            self.zhi = float(zhi)
        else:
            raise ValueError('zhi not found!')

        #the first photometric band
        if 'band1' in list(test_args.keys()):
            band1 = test_args['band1']
            self.band1 = band1
        else:
            raise ValueError('band1 not found!')
        #the second photometric band
        if 'band2' in list(test_args.keys()):
            band2 = test_args['band2']
            self.band2 = band2
        else:
            raise ValueError('band2 not found!')

    def run_validation_test(self, galaxy_catalog, galaxy_catalog_name, output_dict):
        """
        Load galaxy catalog and (re)calculate the color distribution.
        
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
        if not all(k in galaxy_catalog.quantities for k in (self.band1, self.band2)):
            #raise an informative warning
            msg = ('galaxy catalog does not have `band1` quantity, skipping the rest of the validation test.')
            warn(msg)
            #write to log file
            with open(output_dict['log'], 'w') as f:
                f.write(msg)
            
            return 2

        #calculate color distribution in galaxy catalog
        binctr, hist = self.color_distribution(galaxy_catalog)
        catalog_result = (binctr, hist)
        
        #calculate summary statistic
        summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
        
        #plot results
        self.plot_result(catalog_result, galaxy_catalog_name, output_dict['figure'])
        
        #save results to files
        self.write_result_file(catalog_result, output_dict['catalog'])
        self.write_validation_file(self.validation_data, output_dict['validation'])
        self.write_summary_file(summary_result, output_dict['summary'])
            
        return (0 if test_passed else 1)
            
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
        
        #remove nonsensical magnitude values
        # mask = (mag1>0) & (mag1<50) & (mag2>0) & (mag2<50)
        # mag1 = mag1[mask]
        # mag2 = mag2[mask]
        
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
        
        fn = os.path.join(self._data_directory, self._data_file)
        
        binctr, hist = np.loadtxt(fn, unpack=True)
        
        return binctr, hist
    
    def plot_result(self, result, galaxy_catalog_name, savepath):
        """
        Create plot of color distribution
        
        Parameters
        ----------
        result :
            plot of color distribution of mock catalog and observed catalog
        
        galaxy_catalog_name : string
            name of galaxy catalog
        
        savepath : string
            file to save plot
        """
        
        fig = plt.figure()
        
        #plot measurement from galaxy catalog
        mbinctr, mhist = result
        plt.step(mbinctr, mhist, where="mid", label=galaxy_catalog_name, color='blue')
        
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
        binctr, hist = result
        
        #save result to file
        f = open(savepath, 'w')
        if comment:
            f.write('# {0}\n'.format(comment))
        for b, h in zip(binctr, hist):
            f.write("%13.6e %13.6e\n" % (b, h))
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
        

