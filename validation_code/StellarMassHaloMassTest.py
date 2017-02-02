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

__all__ = ['StellarMassHaloMassTest','plot_summary']
__author__ = []

catalog_output_file = 'catalog_smhm.txt'
validation_output_file = 'validation_smhm.txt'
summary_output_file = 'summary_smhm.txt'
summary_details_module ='write_summary_details'
summary_details_file = 'summary_details_smhm.txt'
log_file = 'log_smhm.log'
plot_file = 'plot_smhm.png'
MassiveBlackII = 'MassiveBlackII'
plot_title = 'Average Stellar-mass - Halo-mass Relation'
xaxis_label = '$\log M_{halo}\ (M_\odot)$'
yaxis_label = 'Average $M^{*}\ (M_\odot)$'
summary_colormap = 'rainbow'
test_range_color = 'red'

class StellarMassHaloMassTest(ValidationTest):
    """
    validaton test class object to compute stellar mass function bins
    """
    
    def __init__(self, **kwargs):
        """
        initialize a stellar mass - halo mass function validation test
        
        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots
        
        test_name : string
            string indicating test name
        
        observation : string, optional
            name of stellar-mass-v-halo-mass validation observation:
            MassiveBlackII
        
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
            available_observations = [MassiveBlackII]
            if kwargs['observation'] in available_observations:
                self.observation = kwargs['observation']
            else:
                msg = ('`observation` not available')
                raise ValueError(msg)
        else:
            self.observation = MassiveBlackII
        
        obinctr, mstar_ave, mave_min, mave_max, mstar_min, mstar_max, mstar_up, mstar_dn = self.load_validation_data()
        #bin center, mean, error on mean, lower bound, upper bound, min of bin, max of bin, sigma
        self.validation_data = {'x':obinctr, 'y':mstar_ave, 'y-':mstar_dn, 'y+':mstar_up, 'ymin':mstar_min, 'ymax':mstar_max, 'yup':mstar_up, 'ydn': mstar_dn}
        
        #halo mass bins
        if 'bins' in kwargs:
            self.mhalo_log_bins = np.linspace(*kwargs['bins'])
        else:
            self.mhalo_log_bins = np.linspace(7.0, 15.0, 25)
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
        self.summary_method = kwargs.get('summary','L2Diff')
        self.threshold = kwargs.get('threshold',1.0)
        self.summary_details = kwargs.get('summary_details',False)
        self.validation_range = kwargs.get('validation_range',(8.0,15.0))

    def load_validation_data(self):
        """
        load tabulated stellar mass halo mass function data
        """
        
        #associate files with observations
        stellar_mass_halo_mass_files = {MassiveBlackII:'MASSIVEBLACKII/StellarMass_HaloMass/tab_new.txt',
                                       }
                                      
        #set the columns to use in each file
        columns = {MassiveBlackII:(0,1,3,4,5,6,8,10),
                  }
        #get path to file
        fn = os.path.join(self.base_data_dir, stellar_mass_halo_mass_files[self.observation])
        
        #column 1: halo mass bin center
        #column 2: mean stellar mass
        #column 4: mean stellar mass - error (on mean)
        #column 5: mean stellar mass + error (on mean)
        #column 6: bin minimum
        #column 7: bin maximum
        #column 8: 1-sigma error
        #column 9: 16th percentile 
        #column 11: 84th percentile
        binctr, mstar_ave, mave_min, mave_max, mstar_min, mstar_max, mstar_up, mstar_dn = np.loadtxt(fn, unpack=True, usecols=columns[self.observation])
        
        #take log of values
        binctr = np.log10(binctr)
        mave_max = np.log10(mave_max)
        mave_min = np.log10(mave_min)
        mstar_ave = np.log10(mstar_ave)
        mstar_max = np.log10(mstar_max)
        mstar_min = np.log10(mstar_min)
        mstar_up = np.log10(mstar_up)
        mstar_dn = np.log10(mstar_dn)

        return binctr, mstar_ave, mave_min, mave_max, mstar_min, mstar_max, mstar_dn, mstar_up
    
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
        if not 'stellar_mass' in galaxy_catalog.quantities or not 'mass' in galaxy_catalog.quantities or not 'parent_halo_id' in galaxy_catalog.quantities:
            #raise an informative warning
            msg = ('galaxy catalog does not have either `mass`, `stellar_mass` or `parent_halo_id` quantity, skipping the rest of the validation test.')
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir ,log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return TestResult('SKIPPED', 'missing required quantities: main halos with stellar mass and/or halomass')

        #calculate stellar mass - halo mass function in galaxy catalog
        try:
            binctr, binwid, mhist, mhmin, mhmax = self.profile_stellar_mass_vs_halo_mass(galaxy_catalog)
            catalog_result = {'x':binctr,'dx': binwid, 'y':mhist, 'y-':mhmin, 'y+': mhmax}
        except ValueError as e:
            return TestResult('SKIPPED', '{}'.format(e)[:80])
        
        #calculate summary statistic
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
    
    def profile_stellar_mass_vs_halo_mass(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get stellar masses and halo masses from galaxy catalog
        stellarmasses = galaxy_catalog.get_quantities("stellar_mass", {'zlo': self.zlo, 'zhi': self.zhi})
        halomasses = galaxy_catalog.get_quantities("mass", {'zlo': self.zlo, 'zhi': self.zhi})
        parent_halo_id = galaxy_catalog.get_quantities("parent_halo_id", {'zlo': self.zlo, 'zhi': self.zhi})

        #remove non-finite or negative numbers and select main halos
        mask = np.isfinite(stellarmasses) & (stellarmasses > 0.0) & (parent_halo_id == -1) & np.isfinite(halomasses) & (halomasses > 0.0)
        #check if we have catalog data left
        if (np.sum(mask) ==0):
            msg=('galaxy catalog does not return any valid halos, skipping this validation test.')
            warn(msg) 
            raise ValueError(msg)

        stellarmasses = stellarmasses[mask]
        halomasses = halomasses[mask]

        #bin halo masses in log bins
        logm = np.log10(halomasses)
        mhist, mbins = np.histogram(logm, bins=self.mhalo_log_bins)
        binctr = (mbins[1:] + mbins[:-1])*0.5
        binwid = mbins[1:] - mbins[:-1]

        #compute average stellar mass in each bin
        Nbins=len(mbins)-1
        avg_stellarmass = np.zeros(Nbins)
        avg_stellarmasserr = np.zeros(Nbins)
        for i in range(Nbins):
            binsmass = stellarmasses[(logm >= mbins[i]) & (logm < mbins[i+1])]
            avg_stellarmass[i] = np.mean(binsmass) 
            avg_stellarmasserr[i] = np.std(binsmass)/np.sqrt(len(binsmass))
        avg_stellarmassmin = avg_stellarmass - avg_stellarmasserr
        avg_stellarmassmax = avg_stellarmass + avg_stellarmasserr

        #take log of values
        log_ave_sm=np.log10(avg_stellarmass)
        log_min_sm=np.log10(avg_stellarmassmin)
        log_max_sm=np.log10(avg_stellarmassmax)
        
        return binctr, binwid, log_ave_sm, log_min_sm, log_max_sm

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
        #plot bin extrema for validation data
        #plt.plot(obinctr,self.validation_data['ymin'],label='min',ls='dashed',color='green')
        #plt.plot(obinctr,self.validation_data['ymax'],label='max',ls='dashed',color='green')
        #plt.fill_between(obinctr, self.validation_data['ydn'], self.validation_data['yup'], facecolor='green', alpha=0.3, edgecolor='none')

        #add validation region to plot
        if len(test_details)>0:          #xrange from test_details
            xrange=test_details['x']
        else:                            #xrange from validation_range
            mask = (self.validation_data['x']>self.validation_range[0]) & (self.validation_data['x']<self.validation_range[1])
            xrange = self.validation_data['x'][mask]
        ymin,ymax = plt.gca().get_ylim()
        plt.fill_between(xrange, ymin, ymax, color=test_range_color, alpha=0.15)
        patch = mpatches.Patch(color=test_range_color, alpha=0.1, label='Test Region') #create color patch for legend
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
        if('ymin' in result and 'ymax' in result and 'yup' in result and 'ydn' in result):
            for b, h, hn, hx, hmn, hmx, hdn, hup in zip(*(result[k] for k in ['x','y','y-','y+','ymin','ymax','ydn','yup'])):
                f.write("%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n" % (b, h, hn, hx, hmn, hmx, hdn, hup))
        else:
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
    colors= matplotlib.cm.get_cmap(summary_colormap)(np.linspace(0.,1.,len(catalog_list)))
    
    #loop over catalogs and plot
    for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
        fn = os.path.join(catalog_dir, catalog_output_file)
        sbinctr, shist, shmin, shmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
        plt.step(sbinctr, shist, where="mid", label=catalog_name, color=color)
        plt.fill_between(sbinctr, shmin, shmax, facecolor=color, alpha=0.3, edgecolor='none')
    
    #plot 1 instance of validation data (same for each catalog)
    fn = os.path.join(catalog_dir, validation_output_file)
    obinctr, ohist, ohmin, ohmax, omin, omax, odn, oup = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3,4,5,6,7])
    plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=validation_kwargs['observation'], fmt='o',color='black')
    #plt.plot(obinctr,omin,label='min',ls='dashed',color='black')
    #plt.plot(obinctr,omax,label='max',ls='dashed',color='black')
    #plt.fill_between(obinctr, odn, oup, facecolor='black', alpha=0.3, edgecolor='none')

    plt.legend(loc='best', frameon=False, numpoints=1, fontsize='small')
    
    plt.savefig(output_file)
    plt.close(fig)
