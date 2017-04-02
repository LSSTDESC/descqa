from __future__ import (division, print_function, absolute_import)

import os
import numpy as np
from warnings import warn
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
from astropy import units as u
from ValidationTest import ValidationTest, TestResult
from CalcStats import L2Diff, L1Diff, KS_test
from ComputeColorDistribution import load_SDSS
from scipy.ndimage.filters import uniform_filter1d

catalog_output_file = 'catalog_quantiles.txt'
validation_output_file = 'validation_quantiles.txt'
summary_output_file = 'summary.txt'
log_file = 'log.txt'
plot_pdf_file = 'plot_pdf.png'
plot_cdf_file = 'plot_cdf.png'

class ColorDistributionTest(ValidationTest):
    """
    validaton test class object to compute galaxy color distribution
    """
    
    def __init__(self, load_validation_catalog_q=True, **kwargs):
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

        translate : dictionary, optional
            translate the bands to catalog specific names

        limiting_band: string, required
            band of the magnitude limit in the validation catalog

        limiting_mag: float, required
            the magnitude limit

        zlo : float, requred
            minimum redshift of the validation catalog
        
        zhi : float, requred
            maximum redshift of the validation catalog
                
        data_dir : string, required
            path to the validation data directory

        data_name : string, required
            name of the validation data
        
        load_validation_catalog_q: boolean, optional
            if True, load the full validation catalog and calculate the color distribution
            default: True

        """
        
        super(self.__class__, self).__init__(**kwargs)
        
        # set validation data information
        self._data_dir = kwargs['data_dir']
        self._data_name = kwargs['data_name']
        self._raw_data_fname = kwargs['raw_data_fname']
        
        # set parameters of test
        # colors
        if 'colors' in kwargs:
            self.colors = kwargs['colors']
        else:
            raise ValueError('`colors` not found!')
        for color in self.colors:
            if len(color)!=3 or color[1]!='-':
                raise ValueError('`colors` is not in the correct format!')
        # band of limiting magnitude
        if 'limiting_band' in list(kwargs.keys()):
            self.limiting_band = kwargs['limiting_band']
        else:
            raise ValueError('`limiting_band` not found!')
        # # limiting magnitude
        # if 'limiting_mag' in list(kwargs.keys()):
        #     self.limiting_mag = kwargs['limiting_mag']
        # else:
        #     raise ValueError('`limiting_mag` not found!')
        # limiting k-corrected absolute magnitude
        if 'limiting_abs_mag' in list(kwargs.keys()):
            self.limiting_abs_mag = kwargs['limiting_abs_mag']
        else:
            raise ValueError('`limiting_abs_mag` not found!')

        # Redshift range
        # minimum redshift
        if 'zlo' in list(kwargs.keys()):
            self.zlo_obs = self.zlo_mock = kwargs['zlo']
        else:
            raise ValueError('`zlo` not found!')
        # maximum redshift
        if 'zhi' in list(kwargs.keys()):
            self.zhi_obs = self.zhi_mock = kwargs['zhi']
        else:
            raise ValueError('`zhi` not found!')

        # translation rules from bands to catalog specific names
        if 'translate' in list(kwargs.keys()):
            translate = kwargs['translate']
            self.translate = translate
        else:
            raise ValueError('translate not found!')

        self.load_validation_catalog_q = load_validation_catalog_q

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
        fig_cdf, axes_cdf = plt.subplots(nsubplots, 2, figsize=(11, 4*nsubplots))
        fig_pdf, axes_pdf = plt.subplots(nsubplots, 2, figsize=(11, 4*nsubplots))
        skip_q = True   # False if any color exists in the catalog
        pass_q = True   # False if any color fails
        pass_count = 0   # Number of colors that pass the test

        if not self._data_name in ['SDSS']:
            raise ValueError('Validation data '+self._data_name+'  not found!')

        if hasattr(galaxy_catalog, "SDSS_kcorrection_z"):
            self.SDSS_kcorrection_z = galaxy_catalog.SDSS_kcorrection_z
        else:
            msg = ('galaxy catalog does not have SDSS_kcorrection_z; using default SDSS_kcorrection_z = 0.06\n')
            warn(msg)
            self.SDSS_kcorrection_z = 0.06
            # write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)

        # Cosmololy for distance modulus for absolute magnitudes
        self.cosmology = galaxy_catalog.cosmology

        if self.load_validation_catalog_q:
            # if self._data_name=='DEEP2':
            #     vsummary = load_DEEP2(self._raw_data_fname, self.colors, self.zlo_obs, self.zhi_obs, self.limiting_band, self.limiting_mag)
            if self._data_name=='SDSS':
                vsummary = load_SDSS(self._raw_data_fname, self.colors, self.SDSS_kcorrection_z)

        filename = os.path.join(base_output_dir, summary_output_file)
        with open(filename, 'a') as f:
            f.write('%2.3f < z < %2.3f\n'%(self.zlo_obs, self.zhi_obs))

        # initialize array for quantiles
        catalog_quantiles = np.zeros([len(self.colors), 5])
        validation_quantiles = np.zeros([len(self.colors), 5])
        # loop through colors
        for ax_cdf, ax_pdf, index in zip(axes_cdf.flat, axes_pdf.flat, range(len(self.colors))):

            color = self.colors[index]
            band1 = self.translate[color[0]]
            band2 = self.translate[color[2]]
            self.band1 = band1
            self.band2 = band2

            if self.load_validation_catalog_q:
                obinctr, ohist = vsummary[index]
                ocdf = np.zeros(len(ohist))
                ocdf[0] = ohist[0]
                for cdf_index in range(1, len(ohist)):
                    ocdf[cdf_index] = ocdf[cdf_index-1]+ohist[cdf_index]
            else:
                #load validation summary data
                filename = self._data_name+'_'+color+'_z_%1.3f_%1.3f_pdf.txt'%(self.zlo_obs, self.zhi_obs)
                obinctr, ohist = self.load_validation_data(filename)
                ocdf = np.zeros(len(ohist))
                ocdf[0] = ohist[0]
                for cdf_index in range(1, len(ohist)):
                    ocdf[cdf_index] = ocdf[cdf_index-1]+ohist[cdf_index]

            # #----------------------------------------------------------------------------------------
            # if index==0:
            #    self.validation_data = [(obinctr, ohist)]
            # else:
            #    self.validation_data = self.validation_data + [(obinctr, ohist)]
            # #----------------------------------------------------------------------------------------
            self.validation_data = (obinctr, ocdf)

            # make sure galaxy catalog has appropiate quantities
            if not all(k in galaxy_catalog.quantities for k in (self.band1, self.band2)):
                # raise an informative warning
                msg = ('galaxy catalog does not have `{}` and/or `{}` quantities.\n'.format(band1, band2))
                warn(msg)
                # write to log file
                fn = os.path.join(base_output_dir, log_file)
                with open(fn, 'a') as f:
                    f.write(msg)
                continue

            # calculate color distribution in galaxy catalog
            mbinctr, mhist = self.color_distribution(galaxy_catalog, (-1, 4, 2000), base_output_dir)
            if mbinctr is None:
                # raise an informative warning
                msg = ('The `{}` and/or `{}` quantities don\'t have the correct range or format.\n'.format(band1, band2))
                warn(msg)
                # write to log file
                fn = os.path.join(base_output_dir, log_file)
                with open(fn, 'a') as f:
                    f.write(msg)
                continue

            # At least one color exists
            skip_q = False

            mcdf = np.zeros(len(mhist))
            mcdf[0] = mhist[0]
            for cdf_index in range(1, len(mhist)):
                mcdf[cdf_index] = mcdf[cdf_index-1]+mhist[cdf_index]
            catalog_result = (mbinctr, mhist)

            # 95% and 68% quantiles
            m95min = mbinctr[np.argmax(mcdf>0.025)]
            m95max = mbinctr[np.argmax(mcdf>0.975)]
            o95min = obinctr[np.argmax(ocdf>0.025)]
            o95max = obinctr[np.argmax(ocdf>0.975)]
            m68min = mbinctr[np.argmax(mcdf>0.16)]
            m68max = mbinctr[np.argmax(mcdf>0.84)]
            o68min = obinctr[np.argmax(ocdf>0.16)]
            o68max = obinctr[np.argmax(ocdf>0.84)]
            mmedian = mbinctr[np.argmax(mcdf>0.5)]
            omedian = obinctr[np.argmax(ocdf>0.5)]
            catalog_quantiles[index] = np.array([m95min, m68min, mmedian, m68max, m95max])
            validation_quantiles[index] = np.array([o95min, o68min, omedian, o68max, o95max])

            d1 = {'x':mbinctr, 'y':mcdf}
            d2 = {'x':obinctr, 'y':ocdf}
            d1_shifted = {'x':mbinctr-(mmedian-omedian), 'y':mcdf}
            # calculate L2diff
            L2, L2_success = L2Diff(d1, d2, threshold=0.02)
            L2_shifted, L2_shifted_success = L2Diff(d1_shifted, d2, threshold=0.02)
            # calculate L1Diff
            L1, L1_success = L1Diff(d1, d2)
            L1_shifted, L1_shifted_success = L1Diff(d1_shifted, d2)
            # calculate K-S statistic
            KS, KS_success = KS_test(d1, d2)
            KS_shifted, KS_shifted_success = KS_test(d1_shifted, d2)

            # plot CDF
            # validation distribution
            ax_cdf.step(obinctr, ocdf, label=self._data_name,color='red')
            # catalog distribution
            ax_cdf.step(mbinctr, mcdf, where="mid", label=catalog_name, color='blue')
            # color distribution after constant shift
            ax_cdf.step(mbinctr-(mmedian-omedian), mcdf, where="mid", label=catalog_name+' shifted', linestyle='--', color='blue')
            ax_cdf.set_xlabel(color, fontsize=12)
            ax_cdf.set_title('')
            xmin = np.min([mbinctr[np.argmax(mcdf>0.005)], obinctr[np.argmax(ocdf>0.005)]])
            xmax = np.max([mbinctr[np.argmax(mcdf>0.995)], obinctr[np.argmax(ocdf>0.995)]])
            ax_cdf.set_xlim(xmin, xmax)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.legend(loc='best', frameon=False)

            # plot PDF
            mhist_smooth = uniform_filter1d(mhist, 20)
            ohist_smooth = uniform_filter1d(ohist, 20)
            # validation data
            ax_pdf.step(obinctr, ohist_smooth, label=self._data_name,color='red')
            # catalog distribution
            ax_pdf.step(mbinctr, mhist_smooth, where="mid", label=catalog_name, color='blue')
            # color distribution after constant shift
            ax_pdf.step(mbinctr-(mmedian-omedian), mhist_smooth, where="mid", label=catalog_name+' shifted', linestyle='--', color='blue')
            ax_pdf.set_xlabel(color, fontsize=12)
            ax_pdf.set_xlim(xmin, xmax)
            ax_pdf.set_ylim(ymin=0.)
            ax_pdf.set_title('')
            ax_pdf.legend(loc='best', frameon=False)

            # save result to file
            filename = os.path.join(base_output_dir, summary_output_file)
            with open(filename, 'a') as f:
                f.write("Median "+color+" difference (obs - mock) = %2.3f\n"%(omedian-mmedian))
                f.write(color+" {}: {} = {}\n".format('SUCCESS' if L2_success else 'FAILED', 'L2Diff', L2))
                f.write(color+" (shifted) {}: {} = {}\n".format('SUCCESS' if L2_shifted_success else 'FAILED', 'L2Diff', L2_shifted))
                f.write(color+" {}: {} = {}\n".format('SUCCESS' if L1_success else 'FAILED', 'L1Diff', L1))
                f.write(color+" (shifted) {}: {} = {}\n".format('SUCCESS' if L1_shifted_success else 'FAILED', 'L1Diff', L1_shifted))
                f.write(color+" {}: {} = {}\n".format('SUCCESS' if KS_success else 'FAILED', 'K-S', KS))
                f.write(color+" (shifted) {}: {} = {}\n".format('SUCCESS' if KS_shifted_success else 'FAILED', 'K-S', KS_shifted))

            # The test is considered pass if the all colors pass L2Diff
            if L2_success:
                pass_count+=1
            else:
                pass_q = False

        if not skip_q:
            # save plots
            fn = os.path.join(base_output_dir, plot_cdf_file)
            fig_cdf.savefig(fn)
            fn = os.path.join(base_output_dir, plot_pdf_file)
            fig_pdf.savefig(fn)
        
        plt.close()

        # save quantiles
        fn = os.path.join(base_output_dir, catalog_output_file)
        np.savetxt(fn, catalog_quantiles)
        fn = os.path.join(base_output_dir, validation_output_file)
        np.savetxt(fn, validation_quantiles)

        #--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--
        if skip_q:
            return TestResult(summary='No available colors for comparison. ', skipped=True)
        elif pass_q:
            return TestResult(score=pass_count/float(len(self.colors)), 
                summary='{}/{} success - All colors pass the test. '.format(pass_count, len(self.colors)), passed=True)
        else:
            return TestResult(score=pass_count/float(len(self.colors)), 
                summary='{}/{} success - Not all colors pass the test. '.format(pass_count, len(self.colors)), passed=False)

    def color_distribution(self, galaxy_catalog, bin_args, base_output_dir):
        """
        Calculate the color distribution.
        
        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        
        # get magnitudes from galaxy catalog
        mag1 = galaxy_catalog.get_quantities(self.band1, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        mag2 = galaxy_catalog.get_quantities(self.band2, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})

        if len(mag1)==0:
            msg = 'No object in the redshift range!\n'
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return None, None

        # ############ DEBUG ############
        # limiting_band_name = self.translate[self.limiting_band]        
        # mag_lim = galaxy_catalog.get_quantities(limiting_band_name, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        # print('mag_lim')
        # print(len(mag_lim))
        # print(np.max(mag_lim))
        # print(np.min(mag_lim))
        # print()
        # ############ DEBUG ############

        #apply magnitude limit and remove nonsensical magnitude values
        limiting_band_name = self.translate[self.limiting_band]
        mag_lim = galaxy_catalog.get_quantities(limiting_band_name, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        # mask = (mag_lim<self.limiting_mag) & (mag1>0) & (mag1<50) & (mag2>0) & (mag2<50)
        mask = (mag_lim<self.limiting_abs_mag)
        mag1 = mag1[mask]
        mag2 = mag2[mask]

        if np.sum(mask)==0:
            msg = 'No object in the magnitude range!\n'
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return None, None

        
        # count galaxies
        hist, bins = np.histogram(mag1-mag2, bins=np.linspace(*bin_args))
        # normalize the histogram so that the sum of hist is 1
        hist = hist/np.sum(hist)
        binctr = (bins[1:] + bins[:-1])/2.
        
        return binctr, hist

    def load_validation_data(self, filename):
        """
        Open comparsion validation data, i.e. observational comparison data.
        """
        
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        path = os.path.join(self.base_data_dir, self._data_dir, filename)
        
        binctr, hist = np.loadtxt(path)
        
        return binctr, hist
    
    def write_summary_file(self, result, test_passed, filename, comment=None):
        """
        """
        pass

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
    
    colors = validation_kwargs['colors']
    nsubplots = int(np.ceil(len(colors)/2.))
    fig, axes = plt.subplots(nsubplots, 2, figsize=(11, 6*nsubplots))

    temp = (100*np.ones([len(catalog_list), 5])).tolist()
    # loop over colors
    for ax, index in zip(axes.flat, range(len(colors))):

        # Validation results
        _, catalog_dir = catalog_list[0]
        fn = os.path.join(catalog_dir, validation_output_file)
        vquantiles = np.loadtxt(fn)[index]

        xx = np.linspace(0, len(catalog_list)+1)
        ax.axhline(vquantiles[2], lw=2, color='r', label=validation_kwargs['data_name']+' median')
        ax.axhline(0,xmin=0, xmax=0, lw=7, color='red', alpha=0.3, label=validation_kwargs['data_name']+r' 68%')
        ax.axhline(0,xmin=0, xmax=0, lw=7, color='grey', alpha=0.2, label=validation_kwargs['data_name']+r' 95%')
        ax.fill_between(xx, vquantiles[1], vquantiles[3],facecolor='red', alpha=0.3)
        ax.fill_between(xx, vquantiles[0], vquantiles[1], facecolor='grey', alpha=0.2)
        ax.fill_between(xx, vquantiles[3], vquantiles[4], facecolor='grey', alpha=0.2)

        ymin = vquantiles[0]
        ymax = vquantiles[4]

        # Mock catalog results
        color = colors[index]
        medianprops = dict(color='b')
        D = ax.boxplot(temp, medianprops=medianprops)
        catalog_quantiles = []
        # loop over catalogs and plot
        for catalog_index, (catalog_name, catalog_dir) in zip(range(len(catalog_list)),catalog_list):
            fn = os.path.join(catalog_dir, catalog_output_file)
            quantiles = np.loadtxt(fn)[index]
            if not np.all(quantiles==0):
                D['medians'][catalog_index].set_ydata(quantiles[2])
                D['boxes'][catalog_index]._xy[[0,1,4], 1]=quantiles[1]
                D['boxes'][catalog_index]._xy[[2,3],1]=quantiles[3]
                D['whiskers'][2*catalog_index].set_ydata(np.array([quantiles[1], quantiles[0]]))
                D['whiskers'][2*catalog_index+1].set_ydata(np.array([quantiles[3], quantiles[4]]))
                D['caps'][2*catalog_index].set_ydata(np.array([quantiles[0], quantiles[0]]))
                D['caps'][2*catalog_index+1].set_ydata(np.array([quantiles[4], quantiles[4]]))
                ymin = np.min([ymin, quantiles[0]])
                ymax = np.max([ymax, quantiles[4]])
        ax.set_xlabel('mock catalog')
        ax.set_ylabel(color)

        x = np.arange(1, len(catalog_list)+1)
        labels = [catalog_name for catalog_name, _ in catalog_list]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation='vertical')

        ax.yaxis.grid(True)
        yrange = ymax - ymin
        ax.set_ylim(ymin-0.1*yrange, ymax+0.1*yrange)
        ax.legend(fontsize='small', framealpha=0.4)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
