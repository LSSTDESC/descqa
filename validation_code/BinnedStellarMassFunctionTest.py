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

__all__ = ['BinnedStellarMassFunctionTest', 'plot_summary']

catalog_output_file = 'catalog_smf.txt'
validation_output_file = 'validation_smf.txt'
covariance_matrix_file = 'covariance_smf.txt'
log_file = 'log_smf.txt'
plot_file = 'plot_smf.png'
plot_title = 'Stellar Mass Function'
xaxis_label = r'$\log (M^*/(M_\odot)$'
yaxis_label = r'$dn/dV\,d\log M ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$'
summary_colormap = 'rainbow'
test_range_color = 'red'


class BinnedStellarMassFunctionTest(ValidationTest):
    """
    validation test class object to compute stellar mass function bins
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

        summary_details : boolean, optional

        summary_method : string, optional

        threshold : float, optional

        validation_range : tuple, optional
        """

        super(self.__class__, self).__init__(**kwargs)

        #load validation data
        self.observation = kwargs.get('observation', 'LiWhite2009')
        if self.observation not in ('LiWhite2009', 'MassiveBlackII'):
            raise ValueError('`observation` not available')
        self.validation_data = self.load_validation_data()

        #stellar mass bins
        self.mstar_log_bins = np.array(kwargs.get('bins'))
        if self.mstar_log_bins.size <= 1:
            raise ValueError('`binning` not available or ill defined')

        #redshift range
        self.zlo = float(kwargs.get('zlo', 0.0))
        self.zhi = float(kwargs.get('zhi', 1000.0))

        #statistic options
        self.validation_range = kwargs.get('validation_range', (7.0, 12.0))
        self.summary_statistic = kwargs.get('summary_statistic', 'chisq')
        if self.summary_statistic == 'chisq':
            self.jackknife_nside = int(kwargs.get('jackknife_nside', 5))
        elif self.summary_statistic == 'lpnorm':
            self.jackknife_nside = 0
        else:
            raise ValueError('`summary_statistic` not available')

        #add all other options
        for k in kwargs:
            if not hasattr(self, k):
                setattr(self, k, kwargs[k])


    def load_validation_data(self):
        """
        load tabulated stellar mass function data
        """

        #associate files with observations
        stellar_mass_function_files = {'LiWhite2009':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt',
                                      'MassiveBlackII':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt'}

        #set the columns to use in each file
        columns = {'LiWhite2009':(0,5,6),
                   'MassiveBlackII':(0,1,2),}

        #get path to file
        fn = os.path.join(self.base_data_dir, stellar_mass_function_files[self.observation])

        #column 1: stellar mass bin center
        #column 2: number density
        #column 3: 1-sigma error
        binctr, mhist, merr = np.loadtxt(fn, unpack=True, usecols=columns[self.observation])

        return {'x':np.log10(binctr), 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':np.diag(merr*merr)}


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

        #make sure galaxy catalog has appropriate quantities
        required_quantities = ('stellar_mass', 'positionX', 'positionY', 'positionZ')
        if not all(k in galaxy_catalog.quantities for k in required_quantities):
            #raise an informative warning
            msg = ('galaxy catalog does not have all the required quantities, skipping the rest of the validation test.')
            warn(msg)
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return TestResult(skipped=True)

        #calculate stellar mass function in galaxy catalog
        catalog_result = self.binned_stellar_mass_function(galaxy_catalog)

        #save results to files
        fn = os.path.join(base_output_dir, catalog_output_file)
        self.write_file(catalog_result, fn)

        fn = os.path.join(base_output_dir, validation_output_file)
        self.write_file(self.validation_data, fn)

        fn = os.path.join(base_output_dir, covariance_matrix_file)
        np.savetxt(fn, catalog_result['cov'])

        #plot results
        fn = os.path.join(base_output_dir, plot_file)
        self.plot_result(catalog_result, catalog_name, fn)

        return self.calculate_summary_statistic(catalog_result)


    def binned_stellar_mass_function(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get stellar masses from galaxy catalog
        masses = galaxy_catalog.get_quantities("stellar_mass", {'zlo': self.zlo, 'zhi': self.zhi})
        x = galaxy_catalog.get_quantities("positionX", {'zlo': self.zlo, 'zhi': self.zhi})

        #remove non-finite r negative numbers
        mask  = np.isfinite(masses) 
        mask &= (masses > 0.0)
        mask &= np.isfinite(x)
        
        x = x[mask]
        masses = masses[mask]
        logmasses = np.log10(masses)

        #histogram points and compute bin positions
        mhist, mbins = np.histogram(logmasses, bins=self.mstar_log_bins)
        mhist = mhist.astype(float)
        summass, _   = np.histogram(logmasses, bins=self.mstar_log_bins, weights=masses)
        binctr = (mbins[1:] + mbins[:-1])*0.5
        has_mass = (mhist > 0)
        binctr[has_mass] = np.log10(summass/mhist)[has_mass]
        binwid = mbins[1:] - mbins[:-1]

        #adjust validation range if catlog is missing data in selected validation range
        # nonzerobins=(mhist>0)
        # minlogmass=mbins[np.min(np.where(nonzerobins)[0])]   #find low edge of lowest non-zero bin
        # maxlogmass=mbins[np.max(np.where(nonzerobins)[0])+1] #find hi edge of highest non-zero bin
        # if (self.validation_range[0] < minlogmass):
        #     print('Adjusted lower limit of validation range to match catlog minimum:',minlogmass)
        # if (self.validation_range[1] > maxlogmass):
        #     print('Adjusted upper limit of validation range to match catlog maximum:',maxlogmass)
        # self.validation_range = (max(self.validation_range[0],minlogmass),min(self.validation_range[1],maxlogmass))

        #count galaxies in log bins
        #get errors from jackknife samples if requested
        if self.jackknife_nside > 0:
            y = galaxy_catalog.get_quantities("positionY", {'zlo': self.zlo, 'zhi': self.zhi})[mask]
            z = galaxy_catalog.get_quantities("positionZ", {'zlo': self.zlo, 'zhi': self.zhi})[mask]
            jack_indices = CalcStats.get_subvolume_indices(x, y, z, galaxy_catalog.box_size, self.jackknife_nside)
            njack = self.jackknife_nside**3
            mhist, bias, covariance = CalcStats.jackknife(logmasses, jack_indices, njack, \
                    lambda m, scale: np.histogram(m, bins=self.mstar_log_bins)[0]*scale, \
                    full_args=(1.0,), jack_args=(njack/(njack-1.0),))
            del x, y, z, jack_indices
        else:
            covariance = np.diag(mhist)

        #calculate volume
        if galaxy_catalog.lightcone:
            Vhi = galaxy_catalog.get_cosmology().comoving_volume(self.zhi)
            Vlo = galaxy_catalog.get_cosmology().comoving_volume(self.zlo)
            dV = float((Vhi - Vlo)/u.Mpc**3)
            # TODO: need to consider completeness in volume
            af = float(galaxy_catalog.get_sky_area() / (4.*np.pi*u.sr))
            vol = af * dV
        else:
            vol = galaxy_catalog.box_size**3.0

        #calculate number differential density
        mhist /= (binwid * vol)
        covariance /= (vol*vol)
        covariance /= np.outer(binwid, binwid)
        merr = np.sqrt(np.diag(covariance))

        return {'x':binctr, 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':covariance}


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

        #restrict range of validation data supplied for test if necessary
        mask_validation = (self.validation_data['x'] >= self.validation_range[0]) & (self.validation_data['x'] <= self.validation_range[1])
        mask_catalog = (catalog_result['x'] >= self.validation_range[0]) & (catalog_result['x'] <= self.validation_range[1])
        if np.count_nonzero(mask_validation) != np.count_nonzero(mask_catalog):
            raise ValueError('The length of validation data need to be the same as that of catalog result')

        d = self.validation_data['y'][mask_validation] - catalog_result['y'][mask_catalog]
        nbin = np.count_nonzero(mask_catalog)

        if self.summary_statistic == 'chisq':
            cov = catalog_result['cov'][np.outer(*(mask_catalog,)*2)].reshape(nbin, nbin)
            cov += self.validation_data['cov'][np.outer(*(mask_validation,)*2)].reshape(nbin, nbin)
            score = CalcStats.chisq(d, cov)
            conf_level = getattr(self, 'chisq_conf_level', 0.95)
            score_thres = CalcStats.chisq_threshold(nbin, conf_level)
            passed = score < score_thres
            msg = 'chi^2 = {:g} {} {:g} confidence level = {:g}'.format(score, '<' if passed else '>=', conf_level, score_thres)

        elif self.summary_statistic == 'lpnorm':
            p = getattr(self, 'lpnorm_p', 2.0)
            score = CalcStats.Lp_norm(d, p)
            score_thres = getattr(self, 'lpnorm_thres', 1.0)
            passed = score < score_thres
            msg = 'L{:g} norm = {:g} {} test threshold {:g}'.format(p, score, '<' if passed else '>=', score_thres)

        return TestResult(score, msg, passed)


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
        plt.yscale('log')
        sbinctr, shist, shmin, shmax = (result['x'], result['y'], result['y-'], result['y+'])
        line1, = plt.step(sbinctr, shist, where="mid", label=catalog_name, color='blue')
        plt.fill_between(sbinctr, shmin, shmax, facecolor='blue', alpha=0.3, edgecolor='none')

        #plot comparison data
        obinctr, ohist, ohmin, ohmax = (self.validation_data['x'], self.validation_data['y'], self.validation_data['y-'], self.validation_data['y+'])
        pts1 = plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=self.observation, fmt='o', color='green')

        #add validation region to plot
        mask = (self.validation_data['x'] >= self.validation_range[0]) & (self.validation_data['x'] <= self.validation_range[1])
        xrange = self.validation_data['x'][mask]
        ymin, ymax = plt.gca().get_ylim()
        plt.fill_between(xrange, ymin, ymax, color=test_range_color, alpha=0.15)
        patch = mpatches.Patch(color=test_range_color, alpha=0.1, label='Test Region') #create color patch for legend
        handles = [line1, pts1, patch]

        #add formatting
        plt.legend(handles=handles, loc='best', frameon=False, numpoints=1)
        plt.title(plot_title)
        plt.xlabel(xaxis_label)
        plt.ylabel(yaxis_label)

        #save plot
        fig.savefig(savepath)
        plt.close(fig)


    def write_file(self, result, filename, comment=''):
        """
        write stellar mass function data file

        Parameters
        ----------
        result : dictionary

        filename : string

        comment : string
        """
        np.savetxt(filename, np.vstack((result[k] for k in ('x', 'y', 'y-', 'y+'))).T,
                   fmt='%13.6e', header=comment)



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
    plt.yscale('log')

    #setup colors from colormap
    colors = matplotlib.cm.get_cmap('nipy_spectral')(np.linspace(0., 1., len(catalog_list)))

    #loop over catalogs and plot
    for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
        fn = os.path.join(catalog_dir, catalog_output_file)
        sbinctr, shist, shmin, shmax = np.loadtxt(fn, unpack=True, usecols=(0, 1, 2, 3))
        plt.step(sbinctr, shist, where="mid", label=catalog_name, color=color)
        plt.fill_between(sbinctr, shmin, shmax, facecolor=color, alpha=0.3, edgecolor='none')

    #plot 1 instance of validation data (same for each catalog)
    fn = os.path.join(catalog_dir, validation_output_file)
    obinctr, ohist, ohmin, ohmax = np.loadtxt(fn, unpack=True, usecols=(0, 1, 2, 3))
    plt.errorbar(obinctr, ohist, yerr=[ohist-ohmin, ohmax-ohist], label=validation_kwargs['observation'], fmt='o', color='black')
    plt.legend(loc='best', frameon=False, numpoints=1, fontsize='small')

    plt.savefig(output_file)
    plt.close(fig)
