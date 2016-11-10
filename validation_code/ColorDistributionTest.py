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

        colors : list of string, required
            list of colors to be tested
            e.g ['u-g','g-r','r-i','i-z']

        color_bin_args : list of tuple, required
            list of tuple(minimum color, maximum color, N bins) for each color

        translate : dictionary, optional
            translate the bands to catalog specific names

        limiting_band: string, optional
            band of the magnitude limit in the validation catalog

        limiting_mag: float, optional
            the magnitude limit

        zlo : float, requred
            minimum redshift of the validation catalog
        
        zhi : float, requred
            maximum redshift of the validation catalog
                            
        data_dir : string
            path to the validation data directory
            
        data_name : string
            name of the validation data
        
        test_q: boolean
            if True, zlo and zhi are overwritten with 0 and 1
        """
        
        super(self.__class__, self).__init__(**kwargs)
        
        #set validation data information
        self._data_dir = kwargs['data_dir']
        self._data_name = kwargs['data_name']
        
        #set parameters of test
        #colors
        if 'colors' in kwargs:
            self.colors = kwargs['colors']
        else:
            raise ValueError('`colors` not found!')
        for color in self.colors:
            if len(color)!=3 or color[1]!='-':
                raise ValueError('`colors` is not in the correct format!')
        #color bins
        if 'color_bin_args' in kwargs:
            self.color_bin_args = kwargs['color_bin_args']
        else:
            raise ValueError('`color_bin_args` not found!')
        if len(self.color_bin_args)!=len(self.colors):
            raise ValueError('`colors` and `color_bin_args` should be the same length')
        for color_bin in self.color_bin_args:
            if len(color_bin)!=3:
                raise ValueError('`color_bin_args` is not in the correct format!')
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

        # Redshift range
        if 'test_q' in list(kwargs.keys()) and kwargs['test_q']:
            self.zlo_mock = 0.
            self.zhi_mock = 1.
            self.zlo_obs = kwargs['zlo']
            self.zhi_obs = kwargs['zhi']
        else:
            #minimum redshift
            if 'zlo' in list(kwargs.keys()):
                self.zlo_obs = self.zlo_mock = kwargs['zlo']
            else:
                raise ValueError('`zlo` not found!')
            #maximum redshift
            if 'zhi' in list(kwargs.keys()):
                self.zhi_obs = self.zhi_mock = kwargs['zhi']
            else:
                raise ValueError('`zhi` not found!')

        #translation rules from bands to catalog specific names
        if 'translate' in list(kwargs.keys()):
            translate = kwargs['translate']
            self.translate = translate
        else:
            raise ValueError('translate not found!')



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
        test_passed : boolean
            True if the test is 'passed', False otherwise
        """
        
        nsubplots = int(np.ceil(len(self.colors)/2.))
        fig, ax = plt.subplots(nsubplots, 2, figsize=(14, 5*nsubplots))
        no_data_q = True
        # loop through colors
        for ax1, index in zip(ax.flat, range(len(self.colors))):

            color = self.colors[index]
            band1 = self.translate[color[0]]
            band2 = self.translate[color[2]]
            self.band1 = band1
            self.band2 = band2
            bin_args = self.color_bin_args[index]

            #load validation comparison data
            filename = self._data_name+'_'+color+'_z_%1.3f_%1.3f_bins_%1.2f_%1.2f_%d.txt'%(self.zlo_obs, self.zhi_obs, bin_args[0], bin_args[1], bin_args[2])
            obinctr, ohist = self.load_validation_data(filename)

            # #----------------------------------------------------------------------------------------
            # if index==0:
            #     self.validation_data = [(obinctr, ohist)]
            # else:
            #     self.validation_data = self.validation_data + [(obinctr, ohist)]
            # #----------------------------------------------------------------------------------------
            self.validation_data = (obinctr, ohist)

            #make sure galaxy catalog has appropiate quantities
            if not all(k in galaxy_catalog.quantities for k in (self.band1, self.band2)):
                #raise an informative warning
                msg = ('galaxy catalog does not have {} and/or {} quantity, skipping the rest of the validation test.'.format(band1, band2))
                warn(msg)
                #write to log file
                fn = os.path.join(base_output_dir, log_file)
                with open(fn, 'a') as f:
                    f.write(msg)
                continue

            no_data_q = False
            #calculate color distribution in galaxy catalog
            mbinctr, mhist = self.color_distribution(galaxy_catalog, bin_args, base_output_dir)
            if mbinctr is None:
                return TestResult('SKIPPED', msg='')
            catalog_result = (mbinctr, mhist)

            
            #plot histograms            
            #measurement from galaxy catalog
            ax1.step(mbinctr, mhist, where="mid", label=catalog_name, color='blue')
            #plot validation data
            ax1.step(obinctr, ohist, label=self._data_name,color='green')
            
            #add formatting
            ax1.set_xlabel(color, fontsize=12)
            # ax1.set_title('color distribution')
            ax1.set_title('')
            ax1.set_xlim(bin_args[0], bin_args[1])
            ax1.legend(loc='best', frameon=False)
            #calculate summary statistic
            summary_result, test_passed = self.calulcate_summary_statistic(catalog_result)
            
        #save plot
        if not no_data_q:
            fn = os.path.join(base_output_dir, plot_file)
            fig.savefig(fn)
        
        # #save results to files
        # fn = os.path.join(base_output_dir, catalog_output_file)
        # self.write_file(catalog_result, fn)

        # fn = os.path.join(base_output_dir, validation_output_file)
        # self.write_file(self.validation_data, fn)

        # fn = os.path.join(base_output_dir, summary_output_file)
        # self.write_summary_file(summary_result, fn)
        
        msg = ''
        return TestResult('PASSED' if not no_data_q else 'FAILED', msg)
        # return TestResult('PASSED' if test_passed else 'FAILED', msg)
            
    def color_distribution(self, galaxy_catalog, bin_args, base_output_dir):
        """
        Calculate the color distribution.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get magnitudes from galaxy catalog
        mag1 = galaxy_catalog.get_quantities(self.band1, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        mag2 = galaxy_catalog.get_quantities(self.band2, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})

        if len(mag1)==0:
            warn('No object in the redshift range!')
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return None, None

        if self.limiting_band is not None:
            #apply magnitude limit and remove nonsensical magnitude values
            limiting_band_name = self.translate[self.limiting_band]
            mag_lim = galaxy_catalog.get_quantities(limiting_band_name, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
            print('Applying magnitude limit: '+limiting_band_name+'<%2.2f'%self.limiting_mag)
            mask = (mag_lim<self.limiting_mag) & (mag1>0) & (mag1<50) & (mag2>0) & (mag2<50)
            mag1 = mag1[mask]
            mag2 = mag2[mask]
        else:
            #remove nonsensical magnitude values
            mask = (mag1>0) & (mag1<50) & (mag2>0) & (mag2<50)
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

                    
        #count galaxies
        hist, bins = np.histogram(mag1-mag2, bins=np.linspace(*bin_args))
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
    
    def load_validation_data(self, filename):
        """
        Open comparsion validation data, i.e. observational comparison data.
        """
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path = os.path.join(self.base_data_dir, self._data_dir, filename)
        
        binctr, hist = np.loadtxt(path)
        
        return binctr, hist
    
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
        
    def plot_summary(output_file, test_dicts):
        """
        """
        pass
