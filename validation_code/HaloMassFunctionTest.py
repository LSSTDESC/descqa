"""
"""

from __future__ import (division, print_function, absolute_import)
import numpy as np

from astropy import units as u

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import subprocess

import os
from warnings import warn

from ValidationTest import ValidationTest, TestResult
import CalcStats

__all__ = ['HaloMassFunctionTest','plot_summary']
__author__ = []

catalog_output_file = 'catalog_hmf.txt'
validation_output_file = 'validation_hmf.txt'
summary_output_file = 'summary_hmf.txt'
log_file = 'log_hmf.log'
plot_file = 'plot_hmf.png'
ShethTormen = 'Sheth-Tormen'
Jenkins = 'Jenkins'
Tinker = 'Tinker'
ANALYTIC = 'ANALYTIC'
plot_title = 'Halo-mass Function'
xaxis_label = r'$M_{halo}\ (M_\odot)$'
yaxis_label = r'$dn/dV\, d\logM\ ({\rm Mpc}^{-3})$'
plot_rect = (0.17, 0.15, 0.79, 0.80)
figsize=(4,4)
summary_colormap = 'rainbow'

class HaloMassFunctionTest(ValidationTest):
    """
    validaton test class object to compute stellar mass function bins
    """
    
    def __init__(self, **kwargs):
        """
        initialize a halo mass function validation test
        
        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots
        
        test_name : string
            string indicating test name
        
        observation : string, optional
            name of halo-mass validation observation:
            Sheth-Tormen, Jenkins, Tinker
        
        bins : tuple, optional
        
        zlo : float, optional
        
        zhi : float, optional
        """
        
        super(self.__class__, self).__init__(**kwargs)
        
        #check supplied args
        if 'observation' in kwargs:
            available_observations = [ShethTormen,Jenkins,Tinker]
            if kwargs['observation'] in available_observations:
                self.observation = kwargs['observation']
            else:
                msg = ('`observation` not available')
                raise ValueError(msg)
        else:
            self.observation = ShethTormen

        if 'ztest' in kwargs:
            self.ztest=kwargs['ztest']
        else:
            msg = ('test argument `ztest` not set')
            raise ValueError(msg)
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir ,log_file)
            with open(fn, 'a') as f:
                f.write(msg)
        
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

        self.summary_method = kwargs.get('summary','L2Diff')
        self.threshold = kwargs.get('threshold',1.0)

        #validation data generated in test according to redshift request AND cosmology

    def gen_validation_data(self,ztest,galaxy_catalog,base_output_dir):
        """
        generate halo mass function data
        """
        #associate files with observations
        halo_mass_par = {ShethTormen:'ST',Jenkins:'JEN',Tinker:'TINK',}
        #get path to exe
        exe = os.path.join(self.base_data_dir, 'ANALYTIC/amf/amf.exe')
        fn = os.path.join(base_output_dir, 'analytic.dat')
        inputpars=os.path.join(self.base_data_dir, 'ANALYTIC/amf/input.par')
        subprocess.check_call(["cp", inputpars,base_output_dir])

        #get cosmology from galaxy_catalog
        om = galaxy_catalog.cosmology.Om0
        ob = 0.046 # assume ob is included in om
        h  = galaxy_catalog.cosmology.H(ztest).value/100.
        s8 = 0.816# from paper
        ns = 0.96 # from paper
        #Delta = 700.0 # default from original halo_mf.py
        delta_c = 1.686
        fitting_f = halo_mass_par[self.observation]

        # Example call to amf
        if os.path.exists(fn):
            os.remove(fn)
        args = map(str, [exe, "-omega_0", om, "-omega_bar", ob, "-h", h, "-sigma_8", s8, \
                    "-n_s", ns, "-tf", "EH", "-delta_c", delta_c, "-M_min", 1.0e7, "-M_max", 1.0e15, \
                    "-z", 0.0, "-f", fitting_f])
        print ("Running amf: "+ " ".join(args))
        
        CWD = os.getcwd()
        os.chdir(base_output_dir)
        try:
            with open(os.devnull, 'w') as FNULL:
                p = subprocess.check_call(args, stdout=FNULL, stderr=FNULL)
        finally:
            os.chdir(CWD)

        MassFunc = np.loadtxt(fn).T
        xvals = MassFunc[2]/h
        yvals = MassFunc[3]*h*h*h

        return xvals,yvals

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
            use the TestResult object to return test result
        """
        
        try:
            catalog_result = self.get_galaxy_data(galaxy_catalog)
        except ValueError as e:
            return TestResult('SKIPPED', '{}'.format(e)[:80])

        #generate validation data on the fly according to redshift request AND cosmology
        xvals, yvals = self.gen_validation_data(self.ztest,galaxy_catalog,base_output_dir)
        self.validation_data = {'x':xvals, 'y':yvals}        
        
        
        #calculate summary statistic
        summary_result, test_passed = self.calculate_summary_statistic(catalog_result)
        
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
    
    def get_galaxy_data(self,galaxy_catalog):
        """ 
        get halo mass function                                  
        """
        #make sure galaxy catalog has appropiate quantities
        if not 'mass' in galaxy_catalog.quantities:
            msg = ('galaxy catalog does not have `mass` quantity, skipping the rest of the validation test.')
            warn(msg)
            raise ValueError(msg)

        #calculate stellar mass function in galaxy catalog
        binctr, binwid, mhist, mhmin, mhmax = self.binned_halo_mass_function(galaxy_catalog)
        catalog_result = {'x':binctr,'dx': binwid, 'y':mhist, 'y-':mhmin, 'y+': mhmax}

        return catalog_result

    def binned_halo_mass_function(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        #get halo masses from galaxy catalog
        halomasses = galaxy_catalog.get_quantities("mass", {'zlo': self.zlo, 'zhi': self.zhi})

        #remove non-finite r negative numbers
        mask = np.isfinite(halomasses) & (halomasses > 0.0)
        halomasses = halomasses[mask]
        
        #bin halo masses in log bins
        mhist, mbins = np.histogram(np.log10(halomasses), bins=self.mhalo_log_bins)
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
        binctr=10**binctr
        
        return binctr, binwid, mhist, mhmin, mhmax

    def calculate_summary_statistic(self, catalog_result):
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
        
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes(plot_rect)
        
        #plot comparison data
        obinctr, ohist = (self.validation_data['x'], self.validation_data['y'])
        ax1.loglog(obinctr, ohist, label=self.observation, ls="-", color='green')

        #plot measurement from galaxy catalog
        sbinctr, sbinwid, shist, shmin, shmax = (result['x'], result['dx'], result['y'], result['y-'], result['y+'])
        ax1.errorbar(sbinctr, shist, yerr=[shist-shmin, shmax-shist], ls="none", color='blue', label=catalog_name, marker="o", ms=5)
        
        #add formatting
        plt.legend(loc='best', frameon=False, numpoints=1, fontsize='small')
        plt.grid()
        plt.title(plot_title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)
        
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
        if(not 'y-' in result and not 'y+' in result):
            for b, h, in zip(*(result[k] for k in ['x','y'])):
                f.write("%13.6e %13.6e\n" % (b, h))
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
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes(plot_rect)
    
    #setup colors from colormap
    colors= matplotlib.cm.get_cmap(summary_colormap)(np.linspace(0.,1.,len(catalog_list)))

    #plot 1 instance of validation data (same for each catalog)
    catalog_dir=catalog_list[0][1]   #use first dir
    fn = os.path.join(catalog_dir, validation_output_file)
    obinctr, ohist = np.loadtxt(fn, unpack=True, usecols=[0,1])
    ax1.loglog(obinctr, ohist, label=validation_kwargs['observation'], ls="-", color='black')

    #loop over catalogs and plot
    for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
        fn = os.path.join(catalog_dir, catalog_output_file)
        sbinctr, shist, shmin, shmax = np.loadtxt(fn, unpack=True, usecols=[0,1,2,3])
        ax1.errorbar(sbinctr, shist, yerr=[shist-shmin, shmax-shist], ls="none", color=color, label=catalog_name, marker="o", ms=5)

    plt.legend(loc='best', frameon=False, numpoints=1, fontsize='x-small')
    plt.title(plot_title)
    plt.xlabel(xaxis_label)
    plt.ylabel(yaxis_label)
    plt.grid()

    plt.savefig(output_file)
    
