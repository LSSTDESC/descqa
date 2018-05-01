from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
import numexpr as ne
from astropy.table import Table
from scipy.ndimage.filters import uniform_filter1d
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .stats import CvM_statistic

find_first_true = np.argmax

__all__ = ['ColorDistribution']


# Transformations of DES -> SDSS and DES -> CFHT are derived from Equations A9-12 and
# A19-22 the paper: arxiv.org/abs/1708.01531
# Transformations of SDSS -> CFHT are from:
# www1.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/CFHTLS-SG/docs/extra/filters.html
color_transformation = {'des2sdss': {}, 'des2cfht': {}, 'sdss2cfht': {}}
color_transformation['des2sdss']['g'] = '1.10421 * g - 0.104208 * r'
color_transformation['des2sdss']['r'] = '0.102204 * g + 0.897796 * r'
color_transformation['des2sdss']['i'] = '1.30843 * i - 0.308434 * z'
color_transformation['des2sdss']['z'] = '0.103614 * i + 0.896386 * z'
color_transformation['des2cfht']['g'] = '0.945614 * g + 0.054386 * r'
color_transformation['des2cfht']['r'] = '0.0684211 * g + 0.931579 * r'
color_transformation['des2cfht']['i'] = '1.18646 * i - 0.186458 * z'
color_transformation['des2cfht']['z'] = '0.144792 * i + 0.855208 * z'
color_transformation['sdss2cfht']['u'] = 'u - 0.241 * (u - g)'
color_transformation['sdss2cfht']['g'] = 'g - 0.153 * (g - r)'
color_transformation['sdss2cfht']['r'] = 'r - 0.024 * (g - r)'
color_transformation['sdss2cfht']['i'] = 'i - 0.085 * (r - i)'
color_transformation['sdss2cfht']['z'] = 'z + 0.074 * (i - z)'


class ColorDistribution(BaseValidationTest):
    """
    Compare the mock galaxy color distribution with a validation catalog
    """

    colors = ['u-g', 'g-r', 'r-i', 'i-z']
    summary_output_file = 'summary.txt'
    plot_pdf_file = 'plot_pdf.png'
    plot_cdf_file = 'plot_cdf.png'
    sdss_path = '/global/projecta/projectdirs/lsst/groups/CS/descqa/data/rongpu/SpecPhoto_sdss_mgs_extinction_corrected.fits'
    deep2_path = '/global/projecta/projectdirs/lsst/groups/CS/descqa/data/rongpu/DEEP2_uniq_Terapix_Subaru_trimmed_wights_added.fits'

    def __init__(self, **kwargs): # pylint: disable=W0231

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
            obs_path = self.sdss_path
            obscat = Table.read(obs_path)
            obs_translate = {'u':'modelMag_u', 'g':'modelMag_g', 'r':'modelMag_r', 'i':'modelMag_i', 'z':'modelMag_z'}
            obs_zcol = 'z'
            weights = None
        elif self.validation_catalog == 'DEEP2':
            obs_path = self.deep2_path
            obscat = Table.read(obs_path)
            obs_translate = {'u':'u_apercor', 'g':'g_apercor', 'r':'r_apercor', 'i':'i_apercor', 'z':'z_apercor'}
            obs_zcol = 'zhelio'
            weights = 1/np.array(obscat['p_onmask'])
        else:
            raise ValueError('Validation catalog not recognized')

        # Magnitude and redshift cut
        mask = obscat[obs_translate['r']] < self.obs_r_mag_limit
        mask &= (obscat[obs_zcol] > self.zlo) & (obscat[obs_zcol] < self.zhi)
        obscat = obscat[mask]

        if self.validation_catalog == 'DEEP2':
            # Remove unsecured redshifts
            mask = obscat['zquality'] >= 3
            # Remove CFHTLS-Wide objects
            mask &= obscat['cfhtls_source'] == 0
            obscat = obscat[mask]

        # Selection weights
        if self.validation_catalog == 'SDSS':
            weights = None
        elif self.validation_catalog == 'DEEP2':
            weights = 1/np.array(obscat['p_onmask'])

        # Compute color distribution (PDF, CDF etc.)
        self.obs_color_dist = self.get_color_dist(obscat, obs_translate, weights)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        bands = set(sum((c.split('-') for c in self.colors), []))
        possible_names = ('mag_{}_sdss', 'mag_{}_des', 'mag_true_{}_sdss', 'mag_true_{}_des')
        labels = {band: catalog_instance.first_available(*(n.format(band) for n in possible_names)) for band in bands}
        labels = {k: v for k, v in labels.items() if v}
        if len(labels) < 2:
            return TestResult(skipped=True, summary='magnitudes in mock catalog do not have at least two needed bands.')
        filters = set((v.rpartition('_')[-1] for v in labels.values()))
        if len(filters) > 1:
            return TestResult(skipped=True, summary='magnitudes in mock catalog have mixed filters.')
        filter_this = filters.pop()

        labels['redshift'] = 'redshift_true'
        if not catalog_instance.has_quantity(labels['redshift']):
            return TestResult(skipped=True, summary='mock catalog does not have redshift.')

        # Load mock catalog data
        filters = ['{} > {}'.format(labels['redshift'], self.zlo),
                   '{} < {}'.format(labels['redshift'], self.zhi)]
        data = catalog_instance.get_quantities(list(labels.values()), filters)
        data = {k: data[v] for k, v in labels.items()}

        # Color transformation
        color_trans = None
        if self.color_transformation_q:
            color_trans_name = None
            if self.validation_catalog == 'DEEP2':
                color_trans_name = '{}2cfht'.format(filter_this)
            elif self.validation_catalog == 'SDSS' and filter_this == 'des':
                color_trans_name = 'des2sdss'
            if color_trans_name:
                color_trans = color_transformation[color_trans_name]

        if color_trans:
            data_transformed = {}
            for band in bands:
                try:
                    data_transformed[band] = ne.evaluate(color_trans[band], local_dict=data, global_dict={})
                except KeyError:
                    continue

            data_transformed['redshift'] = data['redshift']
            data = data_transformed
            del data_transformed

        data = GCRQuery('r < {}'.format(self.obs_r_mag_limit)).filter(data)

        # Compute color distribution (PDF, CDF etc.)
        mock_color_dist = self.get_color_dist(data)

        # Calculate Cramer-von Mises statistic
        color_shift = {}
        cvm_omega = {}
        cvm_omega_shift = {}
        for color in self.colors:
            if not ((color in self.obs_color_dist) and (color in mock_color_dist)):
                continue
            color_shift[color] = self.obs_color_dist[color]['median'] - mock_color_dist[color]['median']
            cvm_omega[color] = CvM_statistic(
                mock_color_dist[color]['nsample'], self.obs_color_dist[color]['nsample'],
                mock_color_dist[color]['binctr'], mock_color_dist[color]['cdf'],
                self.obs_color_dist[color]['binctr'], self.obs_color_dist[color]['cdf'])
            cvm_omega_shift[color] = CvM_statistic(
                mock_color_dist[color]['nsample'], self.obs_color_dist[color]['nsample'],
                mock_color_dist[color]['binctr'] + color_shift[color], mock_color_dist[color]['cdf'],
                self.obs_color_dist[color]['binctr'], self.obs_color_dist[color]['cdf'])

        self.make_plots(mock_color_dist, color_shift, cvm_omega, cvm_omega_shift, catalog_name, output_dir)

        # Write to summary file
        fn = os.path.join(output_dir, self.summary_output_file)
        with open(fn, 'a') as f:
            if color_trans:
                f.write('Color transformation: {}\n'.format(color_trans_name))
            else:
                f.write('No color transformation\n')
            f.write('%2.3f < z < %2.3f\n'%(self.zlo, self.zhi))
            f.write('r_mag < %2.3f\n\n'%(self.obs_r_mag_limit))
            for color in self.colors:
                if not ((color in self.obs_color_dist) and (color in mock_color_dist)):
                    continue
                f.write("Median "+color+" difference (obs - mock) = %2.3f\n"%(color_shift[color]))
                f.write(color+": {} = {:2.6f}\n".format('CvM statistic', cvm_omega[color]))
                f.write(color+" (shifted): {} = {:2.6f}\n".format('CvM statistic', cvm_omega_shift[color]))
                f.write("\n")

        return TestResult(inspect_only=True)


    def make_plots(self, mock_color_dist, color_shift, cvm_omega, cvm_omega_shift, catalog_name, output_dir):
        nrows = int(np.ceil(len(self.colors)/2.))
        fig_pdf, axes_pdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))
        fig_cdf, axes_cdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))

        for ax_cdf, ax_pdf, color in zip(axes_cdf.flat, axes_pdf.flat, self.colors):

            if not ((color in self.obs_color_dist) and (color in mock_color_dist)):
                continue

            obinctr = self.obs_color_dist[color]['binctr']
            mbinctr = mock_color_dist[color]['binctr']
            opdf_smooth = self.obs_color_dist[color]['pdf_smooth']
            mpdf_smooth = mock_color_dist[color]['pdf_smooth']
            ocdf = self.obs_color_dist[color]['cdf']
            mcdf = mock_color_dist[color]['cdf']

            xmin = np.min([mbinctr[find_first_true(mcdf > 0.001)],
                           mbinctr[find_first_true(mcdf > 0.001)] + color_shift[color],
                           obinctr[find_first_true(ocdf > 0.001)]])
            xmax = np.max([mbinctr[find_first_true(mcdf > 0.999)],
                           mbinctr[find_first_true(mcdf > 0.999)] + color_shift[color],
                           obinctr[find_first_true(ocdf > 0.999)]])

            # Plot PDF
            # validation data
            ax_pdf.step(obinctr, opdf_smooth, where="mid", label=self.validation_catalog, color='C0')
            # mock color distribution
            ax_pdf.step(mbinctr, mpdf_smooth, where="mid",
                        label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega[color]), color='C1')
            # color distribution after constant shift
            ax_pdf.step(mbinctr + color_shift[color], mpdf_smooth,
                        label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift[color]),
                        linestyle='--', color='C2')
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
                        label=catalog_name+'\n'+r'$\omega={:.3}$'.format(cvm_omega[color]), color='C1')
            # color distribution after constant shift
            ax_cdf.step(mbinctr + color_shift[color], mcdf, where="mid",
                        label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift[color]),
                        linestyle='--', color='C2')
            ax_cdf.set_xlabel('${}$'.format(color))
            ax_cdf.set_title('')
            ax_cdf.set_xlim(xmin, xmax)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.legend(loc='upper left', frameon=False)

        if self.plot_pdf_q:
            fig_pdf.tight_layout()
            fig_pdf.savefig(os.path.join(output_dir, self.plot_pdf_file))
        plt.close(fig_pdf)

        if self.plot_cdf_q:
            fig_cdf.tight_layout()
            fig_cdf.savefig(os.path.join(output_dir, self.plot_cdf_file))
        plt.close(fig_cdf)


    def get_color_dist(self, cat, translate=None, weights=None):
        '''
        Return the color distribution information including PDF, smoothed PDF, and CDF.
        '''
        if translate is None:
            translate = {}

        color_dist = {}
        for color in self.colors:
            band1 = translate.get(color[0], color[0])
            band2 = translate.get(color[-1], color[-1])

            # Remove objects with invalid magnitudes from the analysis
            try:
                cat_mask = (cat[band1] > 0) & (cat[band1] < 50) & (cat[band2] > 0) & (cat[band2] < 50)
            except KeyError:
                continue

            pdf, bin_edges = np.histogram((cat[band1]-cat[band2])[cat_mask],
                                          bins=self.bins,
                                          weights=(None if weights is None else weights[cat_mask]))

            pdf = pdf/np.sum(pdf)
            binctr = (bin_edges[1:] + bin_edges[:-1])/2.
            pdf_smooth = uniform_filter1d(pdf, 20)

            color_dist[color] = {}
            color_dist[color]['nsample'] = np.sum(cat_mask)
            color_dist[color]['binctr'] = binctr
            color_dist[color]['pdf'] = pdf
            color_dist[color]['pdf_smooth'] = pdf_smooth
            color_dist[color]['cdf'] = np.cumsum(pdf)
            color_dist[color]['median'] = np.median((cat[band1]-cat[band2])[cat_mask])

        return color_dist
