from __future__ import division, print_function
import os
from warnings import warn

import numpy as np
from scipy.ndimage.filters import uniform_filter1d

from .ValidationTest import TestResult, plt
from .base import BaseValidationTest
from .CalcStats import CvM_statistic
from .ComputeColorDistribution import load_SDSS

catalog_output_file = 'catalog_quantiles.txt'
validation_output_file = 'validation_quantiles.txt'
summary_output_file = 'summary.txt'
kcorrection_z_file = 'kcorrection_z.txt'
log_file = 'log.txt'
plot_pdf_file = 'plot_pdf.png'
plot_cdf_file = 'plot_cdf.png'
plot_pdf_cdf_file = 'plot_g-r_pdf_cdf.pdf'
data_dir = '/global/cfs/cdirs/lsst/groups/CS/descqa/data/rongpu/'
data_name = 'SDSS'

limiting_band_name = 'SDSS_r:rest:'


find_first_true = np.argmax


class ColorDistributionTest(BaseValidationTest):
    """
    validaton test class object to compute galaxy color distribution
    and compare with SDSS
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

        translate : dictionary, optional
            translate the bands to catalog specific names

        zlo : float, requred
            minimum redshift of the validation catalog

        zhi : float, requred
            maximum redshift of the validation catalog

        threshold : float, required
            threshold value for passing the test
        """

        # set parameters of test:
        # filename of SDSS data
        if 'sdss_fname' in list(kwargs.keys()):
            self.sdss_fname = kwargs['sdss_fname']
        else:
            raise ValueError('`sdss_fname` not found!')
        # colors
        if 'colors' in kwargs:
            self.colors = kwargs['colors']
        else:
            raise ValueError('`colors` not found!')
        for color in self.colors:
            if len(color)!=3 or color[1]!='-':
                raise ValueError('`colors` is not in the correct format!')
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
        # threshold value
        if 'threshold' in list(kwargs.keys()):
            self.threshold = kwargs['threshold']
        else:
            raise ValueError('`threshold` not found!')
        # translation rules from bands to catalog specific names
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
        nrows = int(np.ceil(len(self.colors)/2.))
        fig_cdf, axes_cdf = plt.subplots(nrows, 2, figsize=(8, 4*nrows))
        fig_pdf, axes_pdf = plt.subplots(nrows, 2, figsize=(8, 4*nrows))
        pass_q = True   # False if any color fails
        color_count = 0 # Number of available colors
        pass_count = 0   # Number of colors that pass the test
        cvm_sum = 0.

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

        # Values of the SDSS color distribution histogram
        vsummary, mrmax = load_SDSS(os.path.join(data_dir, self.sdss_fname), self.colors, self.SDSS_kcorrection_z)

        # Write to summary file
        filename = os.path.join(base_output_dir, summary_output_file)
        with open(filename, 'a') as f:
            f.write('K corrected to z = %1.3f\n'%self.SDSS_kcorrection_z)
            f.write('%2.3f < z < %2.3f\n\n'%(self.zlo_obs, self.zhi_obs))
        # Write K correction redshift to file
        filename = os.path.join(base_output_dir, kcorrection_z_file)
        with open(filename, 'a') as f:
            f.write(str(self.SDSS_kcorrection_z))

        # Initialize array for quantiles
        catalog_quantiles = np.zeros([len(self.colors), 5])
        validation_quantiles = np.zeros([len(self.colors), 5])
        # Loop through colors
        for ax_cdf, ax_pdf, index in zip(axes_cdf.flat, axes_pdf.flat, range(len(self.colors))):

            color = self.colors[index]
            band1 = self.translate[color[0]]
            band2 = self.translate[color[2]]
            self.band1 = band1
            self.band2 = band2

            nobs, obinctr, ohist, ocdf = vsummary[index]
            omedian = obinctr[find_first_true(ocdf>0.5)]

            # Make sure galaxy catalog has appropiate quantities
            if not all(k in galaxy_catalog.quantities for k in (self.band1, self.band2)):
                # raise an informative warning
                msg = ('galaxy catalog does not have `{}` and/or `{}` quantities.\n'.format(band1, band2))
                warn(msg)
                # write to log file
                fn = os.path.join(base_output_dir, log_file)
                with open(fn, 'a') as f:
                    f.write(msg)
                continue

            # Calculate color distribution in mock catalog
            color_dist_output = self.color_distribution(galaxy_catalog, (-1, 4, 2000), base_output_dir, omedian, mrmax)
            if color_dist_output is None:
                # raise an informative warning
                msg = ('The `{}` and/or `{}` quantities don\'t have the correct range or format.\n'.format(band1, band2))
                warn(msg)
                # write to log file
                fn = os.path.join(base_output_dir, log_file)
                with open(fn, 'a') as f:
                    f.write(msg)
                continue
            nmock, mbinctr, mhist, mcdf, mhist_shift, mcdf_shift, median_diff = color_dist_output

            # At least one color exists
            color_count += 1

            # Calculate median, quartiles, and 2nd percentile and 98th percentile
            oq1 = obinctr[find_first_true(ocdf>0.25)]
            oq3 = obinctr[find_first_true(ocdf>0.75)]
            oiqr = oq3 - oq1
            # oboxmin = max(oq1-1.5*oiqr, obinctr[find_first_true(ocdf>0)])
            # oboxmax = min(oq3+1.5*oiqr, obinctr[find_first_true(ocdf==ocdf[-1])])
            oboxmin = obinctr[find_first_true(ocdf>0.02)]
            oboxmax = obinctr[find_first_true(ocdf>0.98)]
            mq1 = mbinctr[find_first_true(mcdf>0.25)]
            mq3 = mbinctr[find_first_true(mcdf>0.75)]
            miqr = mq3 - mq1
            mmedian = mbinctr[find_first_true(mcdf>0.5)]
            # mboxmin = max(mq1-1.5*miqr, mbinctr[find_first_true(mcdf>0)])
            # mboxmax = min(mq3+1.5*miqr, mbinctr[find_first_true(mcdf==mcdf[-1])])
            mboxmin = mbinctr[find_first_true(mcdf>0.02)]
            mboxmax = mbinctr[find_first_true(mcdf>0.98)]

            validation_quantiles[index] = np.array([oboxmin, oq1, omedian, oq3, oboxmax])
            catalog_quantiles[index] = np.array([mboxmin, mq1, mmedian, mq3, mboxmax])

            # calculate Cramer-von Mises statistic
            cvm_omega, cvm_success = CvM_statistic(nmock, nobs, mcdf, ocdf, threshold=self.threshold)
            cvm_omega_shift, cvm_success_shift = CvM_statistic(nmock, nobs, mcdf_shift, ocdf, threshold=self.threshold)

            # plot CDF
            # validation distribution
            ax_cdf.step(obinctr, ocdf, label=data_name,color='C1')
            # catalog distribution
            ax_cdf.step(mbinctr, mcdf, where="mid", label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega), color='C0')
            # color distribution after constant shift
            ax_cdf.step(mbinctr, mcdf_shift, where="mid", label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift), linestyle='--', color='C0')
            ax_cdf.set_xlabel('${}$'.format(color))
            ax_cdf.set_title('')
            xmin = np.min([mbinctr[find_first_true(mcdf>0.001)], mbinctr[find_first_true(mcdf_shift>0.001)], obinctr[find_first_true(ocdf>0.001)]])
            xmax = np.max([mbinctr[find_first_true(mcdf>0.999)], mbinctr[find_first_true(mcdf_shift>0.999)], obinctr[find_first_true(ocdf>0.999)]])
            ax_cdf.set_xlim(xmin, xmax)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.legend(loc='upper left', frameon=False)

            # plot PDF
            ohist_smooth = uniform_filter1d(ohist, 20)
            mhist_smooth = uniform_filter1d(mhist, 20)
            mhist_shift_smooth = uniform_filter1d(mhist_shift, 20)
            # validation data
            ax_pdf.step(obinctr, ohist_smooth, label=data_name,color='C1')
            # catalog distribution
            ax_pdf.step(mbinctr, mhist_smooth, where="mid", label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega), color='C0')
            # color distribution after constant shift
            ax_pdf.step(mbinctr, mhist_shift_smooth, where="mid", label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift), linestyle='--', color='C0')
            ax_pdf.set_xlabel('${}$'.format(color))
            ax_pdf.set_xlim(xmin, xmax)
            ax_pdf.set_ylim(ymin=0.)
            ax_pdf.set_title('')
            ax_pdf.legend(loc='upper left', frameon=False)

            # PDF+CDF plot for the paper
            if index==1:    # g-r color
                fig_pdf_cdf, axes_pdf_cdf = plt.subplots(1, 2, figsize=(8, 4))
                axes_pdf_cdf[0].step(obinctr, ohist_smooth, label=data_name,color='C1')
                axes_pdf_cdf[0].step(mbinctr, mhist_smooth, where="mid", label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega), color='C0')
                axes_pdf_cdf[0].step(mbinctr, mhist_shift_smooth, where="mid", label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift), linestyle='--', color='C0')
                axes_pdf_cdf[0].set_xlabel('${}$'.format(color))
                axes_pdf_cdf[0].set_xlim(xmin, xmax)
                axes_pdf_cdf[0].set_ylim(ymin=0.)
                axes_pdf_cdf[0].legend(loc='upper left', frameon=False)
                axes_pdf_cdf[1].step(obinctr, ocdf, label=data_name,color='C1')
                axes_pdf_cdf[1].step(mbinctr, mcdf, where="mid", label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega), color='C0')
                axes_pdf_cdf[1].step(mbinctr, mcdf_shift, where="mid", label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift), linestyle='--', color='C0')
                axes_pdf_cdf[1].set_xlabel('${}$'.format(color))
                axes_pdf_cdf[1].set_xlim(xmin, xmax)
                axes_pdf_cdf[1].set_ylim(0, 1)
                axes_pdf_cdf[1].legend(loc='upper left', frameon=False)
                fn = os.path.join(base_output_dir, plot_pdf_cdf_file)
                fig_pdf_cdf.tight_layout()
                fig_pdf_cdf.savefig(fn)
                plt.close(fig_pdf_cdf)

            # save result to file
            filename = os.path.join(base_output_dir, summary_output_file)
            with open(filename, 'a') as f:
                f.write("Median "+color+" difference (mock - obs) = %2.3f\n"%(median_diff))
                f.write(color+" {}: {} = {}\n".format('SUCCESS' if cvm_success else 'FAILED', 'CvM statistic', cvm_omega))
                f.write(color+" (shifted) {}: {} = {}\n".format('SUCCESS' if cvm_success_shift else 'FAILED', 'CvM statistic', cvm_omega_shift))
                f.write("\n")

            # The test is considered pass if the all colors pass L2Diff
            if cvm_success_shift:
                pass_count+=1
            else:
                pass_q = False

            cvm_sum += cvm_omega_shift
            color_count += 1

        if color_count>0:
            # save plots
            fig_cdf.tight_layout()
            fn = os.path.join(base_output_dir, plot_cdf_file)
            fig_cdf.savefig(fn)
            fig_pdf.tight_layout()
            fn = os.path.join(base_output_dir, plot_pdf_file)
            fig_pdf.savefig(fn)

        plt.close(fig_cdf)
        plt.close(fig_pdf)

        # save quantiles
        fn = os.path.join(base_output_dir, catalog_output_file)
        np.savetxt(fn, catalog_quantiles)
        fn = os.path.join(base_output_dir, validation_output_file)
        np.savetxt(fn, validation_quantiles)

        #--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--#--
        if color_count>0:
            cvm_omega_average = cvm_sum/color_count
        if color_count==0:
            return TestResult(summary='No available colors for comparison. ', skipped=True)
        elif pass_q:
            return TestResult(score=cvm_omega_average,
                              summary='{}/{} success - all colors pass the test; average Cramer-von Mises statistic = {:.3f}'.format(pass_count, len(self.colors), cvm_omega_average), passed=True)
        else:
            return TestResult(score=cvm_omega_average,
                summary='{}/{} success - not all colors pass the test; average CvM statistic = {:.3f}'.format(pass_count, len(self.colors), cvm_omega_average), passed=False)

    def color_distribution(self, galaxy_catalog, bin_args, base_output_dir, omedian, mrmax):
        """
        Calculate the color distribution of mock catalog.

        Parameters
        ----------
        galaxy_catalog : (mock) galaxy catalog reader object
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
            return

        #apply magnitude limit and remove nonsensical magnitude values
        mag_lim = galaxy_catalog.get_quantities(limiting_band_name, {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        mask = (mag_lim<mrmax)

        # # r-band apparent magnitude cut
        # r_band_mag = galaxy_catalog.get_quantities('SDSS_r:observed:', {'zlo': self.zlo_mock, 'zhi': self.zhi_mock})
        # mask = mask & (r_band_mag<17.77)

        mag1 = mag1[mask]
        mag2 = mag2[mask]


        if np.sum(mask)==0:
            msg = 'No object in the magnitude range!\n'
            warn(msg)
            #write to log file
            fn = os.path.join(base_output_dir, log_file)
            with open(fn, 'a') as f:
                f.write(msg)
            return

        mmedian = np.median(mag1-mag2)
        median_diff = mmedian - omedian

        # Histrogram
        hist, bins = np.histogram(mag1-mag2, bins=np.linspace(*bin_args))
        hist_shift, _ = np.histogram(mag1-mag2-median_diff, bins=np.linspace(*bin_args))
        # normalize the histogram so that the sum of hist is 1
        hist = hist/np.sum(hist)
        hist_shift = hist_shift/np.sum(hist_shift)
        binctr = (bins[1:] + bins[:-1])/2.
        # Convert PDF to CDF
        cdf = np.zeros(len(hist))
        cdf[0] = hist[0]
        for cdf_index in range(1, len(hist)):
            cdf[cdf_index] = cdf[cdf_index-1]+hist[cdf_index]
        cdf_shift = np.zeros(len(hist_shift))
        cdf_shift[0] = hist_shift[0]
        for cdf_index in range(1, len(hist_shift)):
            cdf_shift[cdf_index] = cdf_shift[cdf_index-1]+hist_shift[cdf_index]

        return len(mag1), binctr, hist, cdf, hist_shift, cdf_shift, median_diff

    def plot_summary(self, output_file, catalog_list):
        """
        make summary plot for validation test

        Parameters
        ----------
        output_file: string
            filename for summary plot

        catalog_list: list of tuple
            list of (catalog, catalog_output_dir) used for each catalog comparison
        """

        colors = self.colors
        nrows = int(np.ceil(len(colors)/2.))
        fig, axes = plt.subplots(nrows, 2, figsize=(8, 4.*nrows), sharex=True)

        # Sort catalogs by kcorrect_z and names
        catalog_names = [catalog_name for catalog_name, _ in catalog_list]
        argsort = np.argsort(catalog_names, kind='mergesort')
        catalog_list = [catalog_list[i] for i in argsort]
        kcorrection_z_list = []
        for catalog_name, catalog_dir in catalog_list:
            fn = os.path.join(catalog_dir, kcorrection_z_file)
            kcorrection_z_list.append(float(np.loadtxt(fn)))
        argsort = np.argsort(kcorrection_z_list, kind='mergesort')
        catalog_list = [catalog_list[i] for i in argsort]

        # Load summary quantiles data
        data = []
        for _, catalog_dir in catalog_list:
            fn = os.path.join(catalog_dir, catalog_output_file)
            data.append(np.loadtxt(fn))
        data = np.array(data)
        data[~data.any(axis=-1)] = np.nan # to hide catalogs that do not have all colors

        # loop over colors
        for index, ax in enumerate(axes.flat):
            if index >= len(colors):
                ax.axis('off')
                continue

            # Mock catalog results
            ax.boxplot(data[:,index].T, whis='range', medianprops=dict(color='k'))
            ax.set_ylabel('${}$'.format(colors[index]), fontsize=16)

            # Validation results
            first_plot = True
            for cat_index in range(len(catalog_list)):
                _, catalog_dir = catalog_list[cat_index]
                fn = os.path.join(catalog_dir, validation_output_file)
                vquantiles = np.loadtxt(fn)[index]
                if not np.all(vquantiles==0):
                    # xmin and xmax are relative coordinates in range of 0-1.
                    xmin, xmax = [cat_index/len(catalog_list), (cat_index+1)/len(catalog_list)]
                    if first_plot:
                        ax.axhline(vquantiles[2], xmin=xmin, xmax=xmax, lw=2, color='r', label='{} median'.format(data_name))
                        ax.axhspan(vquantiles[1], vquantiles[3], xmin=xmin, xmax=xmax, facecolor='r', alpha=0.3, lw=0, label=' [$Q_1$, $Q_3$]')
                        ax.axhspan(vquantiles[0], vquantiles[1], xmin=xmin, xmax=xmax, facecolor='grey', alpha=0.2, lw=0, label=' [2nd, 98th]')
                        ax.axhspan(vquantiles[3], vquantiles[4], xmin=xmin, xmax=xmax, facecolor='grey', alpha=0.2, lw=0)
                        first_plot = False
                    else:
                        ax.axhline(vquantiles[2], xmin=xmin, xmax=xmax, lw=2, color='r')
                        ax.axhspan(vquantiles[1], vquantiles[3], xmin=xmin, xmax=xmax, facecolor='r', alpha=0.3, lw=0)
                        ax.axhspan(vquantiles[0], vquantiles[1], xmin=xmin, xmax=xmax, facecolor='grey', alpha=0.2, lw=0)
                        ax.axhspan(vquantiles[3], vquantiles[4], xmin=xmin, xmax=xmax, facecolor='grey', alpha=0.2, lw=0)

            x = np.arange(1, len(catalog_list)+1)
            labels = [catalog_name for catalog_name, _ in catalog_list]
            ax.set_xticks(x)
            ax.set_xticks([], True)
            if index >= (axes.size - 2):
                ax.set_xticklabels(labels, rotation='vertical')
            else:
                ax.set_xticklabels(['' for _ in x])

            ax.yaxis.grid(True)
            #ymin = min(vquantiles[0], data[:,index,0].min())
            #ymax = max(vquantiles[4], data[:,index,4].max())
            #yrange = ymax - ymin
            #ax.set_ylim(ymin-0.15*yrange, ymax+0.15*yrange)
            if index==3:
                ax.legend(fontsize='small', framealpha=0.4, loc='lower right')

        plt.tight_layout()
        plt.savefig(output_file)
        plt.savefig(output_file+'.pdf')
        plt.close()

