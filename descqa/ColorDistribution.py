from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .plotting import plt
from astropy.table import Table
from scipy.ndimage.filters import uniform_filter1d

import GCRCatalogs
from .base import BaseValidationTest, TestResult
from .stats import CvM_statistic

colors = ['u-g', 'g-r', 'r-i', 'i-z']
summary_output_file = 'summary.txt'
plot_pdf_file = 'plot_pdf.png'
plot_cdf_file = 'plot_cdf.png'
sdss_path = '/global/projecta/projectdirs/lsst/groups/CS/descqa/data/rongpu/SpecPhoto_sdss_mgs_extinction_corrected.fits'
deep2_path = '/global/projecta/projectdirs/lsst/groups/CS/descqa/data/rongpu/DEEP2_uniq_Terapix_Subaru_trimmed_wights_added.fits'


find_first_true = np.argmax

__all__ = ['ColorDistribution']

class ColorDistribution(BaseValidationTest):
    """
    Compare the mock galaxy color distribution with a validation catalog
    """
    def __init__(self, **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.obs_r_mag_limit = kwargs['obs_r_mag_limit']
        self.zlo = kwargs['zlo']
        self.zhi = kwargs['zhi']
        self.validation_catalog = kwargs['validation_catalog']
        self.plot_pdf_q = kwargs.get('plot_pdf_q', True)
        self.plot_cdf_q = kwargs.get('plot_cdf_q', True)
        self.color_transformation_q = kwargs.get('color_transformation_q', True)

        # bins of color distribution
        self.bins = np.linspace(-1, 4, 2000)
        self.binsize = self.bins[1] - self.bins[0]

        # Load validation catalog and define catalog-specific properties
        if self.validation_catalog == 'SDSS':
            obs_path = sdss_path
            obscat = Table.read(obs_path)
            obs_translate = {'u':'modelMag_u', 'g':'modelMag_g', 'r':'modelMag_r', 'i':'modelMag_i', 'z':'modelMag_z'}
            obs_zcol = 'z'
            weights = None
        elif self.validation_catalog == 'DEEP2':
            obs_path = deep2_path
            obscat = Table.read(obs_path)
            obs_translate = {'u':'u_apercor', 'g':'g_apercor', 'r':'r_apercor', 'i':'i_apercor', 'z':'z_apercor'}
            obs_zcol = 'zhelio'
            weights = 1/np.array(obscat['p_onmask'])
        else:
            raise ValueError('Validation catalog not recognized')
        
        # Magnitude and redshift cut
        mask = obscat[obs_translate['r']] < self.obs_r_mag_limit
        mask &= (obscat[obs_zcol]>self.zlo) & (obscat[obs_zcol]<self.zhi)
        obscat = obscat[mask]

        # Remove unsecure redshifts from DEEP2
        if self.validation_catalog == 'DEEP2':
            mask = obscat['zquality']>=3
            obscat = obscat[mask]

        # Selection weights
        if self.validation_catalog == 'SDSS':
            weights = None
        elif self.validation_catalog == 'DEEP2':
            weights = 1/np.array(obscat['p_onmask'])
        # Compute color distribution (PDF, CDF etc.)
        self.obs_color_dist = self.get_color_dist(colors, obs_translate, obscat, weights)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # # check if needed quantities exist
        # if not catalog_instance.has_quantities(['ra', 'dec']):
        #     return TestResult(skipped=True, summary='do not have needed quantities')

        self.catalog_name = catalog_name
        self.output_dir = output_dir

        # catalog-specific properties
        if catalog_name.startswith('protoDC2'):
            mock_translate_original = {'u':'mag_u_sdss', 'g':'mag_g_sdss', 'r':'mag_r_sdss', 'i':'mag_i_sdss', 'z':'mag_z_sdss'}
            mock_zcol = 'redshift_true'
        elif catalog_name.startswith('buzzard'):
            mock_translate_original = {'g':'mag_g_des', 'r':'mag_r_des', 'i':'mag_i_des', 'z':'mag_z_des'}
            mock_zcol = 'redshift_true'
        else:
            return TestResult(skipped=True, summary='Mock catalog not recognized')

        # Load mock catalog
        gc = GCRCatalogs.load_catalog(catalog_name)
        data = Table(gc.get_quantities([mock_zcol] + list(mock_translate_original.values())))

        # Color transformation
        if self.color_transformation_q:
            data, mock_translate = self.color_transformation(data, self.validation_catalog, self.catalog_name, mock_translate_original)
        else:
            mock_translate = mock_translate_original

        # Magnitude and redshift cut
        mask = data[mock_translate['r']] < self.obs_r_mag_limit
        mask &= ((data[mock_zcol]>self.zlo) & (data[mock_zcol]<self.zhi))
        data = data[mask]

        # Compute color distribution (PDF, CDF etc.)
        self.mock_color_dist = self.get_color_dist(colors, mock_translate, data)

        # Calculate Cramer-von Mises statistic
        self.color_shift = {}
        self.cvm_omega = {}
        self.cvm_omega_shift = {}
        for color in colors:
            if not ((color in self.obs_color_dist) and (color in self.mock_color_dist)):
                continue
            self.color_shift[color] = self.obs_color_dist[color]['median'] - self.mock_color_dist[color]['median']
            self.cvm_omega[color] = CvM_statistic(
                self.mock_color_dist[color]['nsample'], self.obs_color_dist[color]['nsample'], 
                self.mock_color_dist[color]['binctr'], self.mock_color_dist[color]['cdf'], 
                self.obs_color_dist[color]['binctr'], self.obs_color_dist[color]['cdf'])
            self.cvm_omega_shift[color] = CvM_statistic(
                self.mock_color_dist[color]['nsample'], self.obs_color_dist[color]['nsample'], 
                self.mock_color_dist[color]['binctr']+self.color_shift[color], self.mock_color_dist[color]['cdf'], 
                self.obs_color_dist[color]['binctr'], self.obs_color_dist[color]['cdf'])

        self.make_plots()

        # Write to summary file
        fn = os.path.join(self.output_dir, summary_output_file)
        with open(fn, 'a') as f:
            f.write('%2.3f < z < %2.3f\n'%(self.zlo, self.zhi))
            f.write('r_mag < %2.3f\n\n'%(self.obs_r_mag_limit))
            for color in colors:
                if not ((color in self.obs_color_dist) and (color in self.mock_color_dist)):
                    continue
                f.write("Median "+color+" difference (obs - mock) = %2.3f\n"%(self.color_shift[color]))
                f.write(color+": {} = {:2.6f}\n".format('CvM statistic', self.cvm_omega[color]))
                f.write(color+" (shifted): {} = {:2.6f}\n".format('CvM statistic', self.cvm_omega_shift[color]))
                f.write("\n")

        return TestResult(inspect_only=True)

    def make_plots(self):
        nrows = int(np.ceil(len(colors)/2.))
        fig_pdf, axes_pdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))
        fig_cdf, axes_cdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))

        for ax_cdf, ax_pdf, index in zip(axes_cdf.flat, axes_pdf.flat, range(len(colors))):

            color = colors[index]
            if not ((color in self.obs_color_dist) and (color in self.mock_color_dist)):
                continue

            obinctr = self.obs_color_dist[color]['binctr']
            mbinctr = self.mock_color_dist[color]['binctr']
            opdf_smooth = self.obs_color_dist[color]['pdf_smooth']
            mpdf_smooth = self.mock_color_dist[color]['pdf_smooth']
            ocdf = self.obs_color_dist[color]['cdf']
            mcdf = self.mock_color_dist[color]['cdf']

            xmin = np.min([mbinctr[find_first_true(mcdf>0.001)], 
                           mbinctr[find_first_true(mcdf>0.001)] + self.color_shift[color], 
                           obinctr[find_first_true(ocdf>0.001)]])
            xmax = np.max([mbinctr[find_first_true(mcdf>0.999)], 
                           mbinctr[find_first_true(mcdf>0.999)] + self.color_shift[color], 
                           obinctr[find_first_true(ocdf>0.999)]])

            # Plot PDF
            # validation data
            ax_pdf.step(obinctr, opdf_smooth, where="mid", label=self.validation_catalog, color='C0')
            # mock color distribution
            ax_pdf.step(mbinctr, mpdf_smooth, where="mid", 
                label=self.catalog_name+'\n'+r'$\omega={:.3}$'.format(self.cvm_omega[color]), color='C1')
            # color distribution after constant shift
            ax_pdf.step(mbinctr + self.color_shift[color], mpdf_smooth, 
                label=self.catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(self.cvm_omega_shift[color]), linestyle='--', color='C2')
            ax_pdf.set_xlabel('${}$'.format(color))
            ax_pdf.set_xlim(xmin, xmax)
            ax_pdf.set_ylim(ymin=0.)
            ax_pdf.set_title('')
            ax_pdf.legend(loc='upper left', frameon=False)

            # Plot CDF
            # validation distribution
            ax_cdf.step(obinctr, ocdf, label=self.validation_catalog, color='C0')
            # catalog distribution
            ax_cdf.step(mbinctr, mcdf, where="mid", 
                label=self.catalog_name+'\n'+r'$\omega={:.3}$'.format(self.cvm_omega[color]), color='C1')
            # color distribution after constant shift
            ax_cdf.step(mbinctr + self.color_shift[color], mcdf, where="mid", 
                label=self.catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(self.cvm_omega_shift[color]), linestyle='--', color='C2')
            ax_cdf.set_xlabel('${}$'.format(color))
            ax_cdf.set_title('')
            ax_cdf.set_xlim(xmin, xmax)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.legend(loc='upper left', frameon=False)

        if self.plot_pdf_q:
            fig_pdf.tight_layout()
            fig_pdf.savefig(os.path.join(self.output_dir, plot_pdf_file))
        plt.close(fig_pdf)

        if self.plot_cdf_q:
            fig_cdf.tight_layout()
            fig_cdf.savefig(os.path.join(self.output_dir, plot_cdf_file))
        plt.close(fig_cdf)

    def get_color_dist(self, colors, translate, cat, weights=None):
        '''
        Return the color distribution information including PDF, smoothed PDF, and CDF.
        '''
        color_dist = {}

        for color in colors:

            if (color[0] in translate) and (color[2] in translate):
                band1 = translate[color[0]]
                band2 = translate[color[2]]
            else:
                continue

            # Remove objects with invalid magnitudes from the analysis
            cat_mask = (cat[band1]>0) & (cat[band1]<50) & (cat[band2]>0) & (cat[band2]<50)

            if weights is None:
                pdf, bin_edges = np.histogram((cat[band1]-cat[band2])[cat_mask], 
                    bins=self.bins, weights=None)
            else:
                pdf, bin_edges = np.histogram((cat[band1]-cat[band2])[cat_mask], 
                    bins=self.bins, weights=weights[cat_mask])
            pdf = pdf/np.sum(pdf)
            binctr = (bin_edges[1:] + bin_edges[:-1])/2.

            # Convert PDF to CDF
            cdf = np.zeros(len(pdf))
            cdf[0] = pdf[0]
            for cdf_index in range(1, len(pdf)):
                cdf[cdf_index] = cdf[cdf_index-1]+pdf[cdf_index]

            pdf_smooth = uniform_filter1d(pdf, 20)

            color_dist[color] = {}
            color_dist[color]['nsample'] = np.sum(cat_mask)
            color_dist[color]['binctr'] = binctr
            color_dist[color]['pdf'] = pdf
            color_dist[color]['pdf_smooth'] = pdf_smooth
            color_dist[color]['cdf'] = cdf
            color_dist[color]['median'] = np.median((cat[band1]-cat[band2])[cat_mask])

        return color_dist


    def color_transformation(self, data, validation_name, mock_name, mock_translate_original):
        '''
        Return the mock catalog (data) with new columns of transformed magnitudes.

        Transformations of DES -> SDSS and DES -> CFHT are derived from Equations A9-12 and
        A19-22 the paper: arxiv.org/abs/1708.01531

        Transformations of SDSS -> CFHT are from:
        www1.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/CFHTLS-SG/docs/extra/filters.html
        '''
        if (validation_name=='SDSS') and (mock_name.startswith('buzzard')):
            # DES -> SDSS transformation
            data['mag_g_sdss'] = 1.10421 * data['mag_g_des'] - 0.104208 * data['mag_r_des']
            data['mag_r_sdss'] = 0.102204 * data['mag_g_des'] + 0.897796 * data['mag_r_des']
            data['mag_i_sdss'] = 1.30843 * data['mag_i_des'] - 0.308434 * data['mag_z_des']
            data['mag_z_sdss'] = 0.103614 * data['mag_i_des'] + 0.896386 * data['mag_z_des']
            translate = {'g':'mag_g_sdss', 'r':'mag_r_sdss', 'i':'mag_i_sdss', 'z':'mag_z_sdss'}
        elif (validation_name=='DEEP2') and (mock_name.startswith('buzzard')):
            # DES -> CFHT transformation
            data['mag_g_cfht'] = 0.945614 * data['mag_g_des'] + 0.054386 * data['mag_r_des']
            data['mag_r_cfht'] = 0.0684211 * data['mag_g_des'] + 0.931579 * data['mag_r_des']
            data['mag_i_cfht'] = 1.18646 * data['mag_i_des'] - 0.186458 * data['mag_z_des']
            data['mag_z_cfht'] = 0.144792 * data['mag_i_des'] + 0.855208 * data['mag_z_des']
            translate = {'g':'mag_g_cfht', 'r':'mag_r_cfht', 'i':'mag_i_cfht', 'z':'mag_z_cfht'}
        elif (validation_name=='SDSS') and (mock_name.startswith('protoDC2')):
            # Same filters, so no transformation needed
            translate = mock_translate_original
        elif (validation_name=='DEEP2') and (mock_name.startswith('protoDC2')):
            # SDSS -> CFHT (MegaCam) transformation
            data['mag_u_cfht'] = data['mag_u_sdss'] - 0.241 * (data['mag_u_sdss'] - data['mag_g_sdss'])
            data['mag_g_cfht'] = data['mag_g_sdss'] - 0.153 * (data['mag_g_sdss'] - data['mag_r_sdss'])
            data['mag_r_cfht'] = data['mag_r_sdss'] - 0.024 * (data['mag_g_sdss'] - data['mag_r_sdss'])
            data['mag_i_cfht'] = data['mag_i_sdss'] - 0.085 * (data['mag_r_sdss'] - data['mag_i_sdss'])
            data['mag_z_cfht'] = data['mag_z_sdss'] + 0.074 * (data['mag_i_sdss'] - data['mag_z_sdss'])
            translate = {'u':'mag_u_cfht', 'g':'mag_g_cfht', 'r':'mag_r_cfht', 'i':'mag_i_cfht', 'z':'mag_z_cfht'}
        else:
            raise ValueError('Color transformation between of validation and mock catalog not defined')

        return data, translate