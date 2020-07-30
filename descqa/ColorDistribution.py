from __future__ import unicode_literals, absolute_import, division
import os
import re
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
color_transformation = {'des2sdss': {}, 'des2cfht': {}, 'sdss2cfht': {}, 'lsst2cfht': {}, 'lsst2sdss':{}}
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

    def __init__(self, **kwargs): # pylint: disable=W0231

        # load test config options
        self.kwargs = kwargs
        self.obs_r_mag_limit = kwargs.get('obs_r_mag_limit', None)
        self.lightcone = kwargs.get('lightcone', True)
        if self.lightcone:
            self.zlo = kwargs['zlo']
            self.zhi = kwargs['zhi']
        self.validation_catalog = kwargs.get('validation_catalog', None)
        self.plot_pdf_q = kwargs.get('plot_pdf_q', True)
        self.plot_cdf_q = kwargs.get('plot_cdf_q', True)
        self.color_transformation_q = kwargs.get('color_transformation_q', True)
        self.Mag_r_limit = kwargs.get('Mag_r_limit', None)
        self.rest_frame = kwargs.get('rest_frame', bool(self.Mag_r_limit and not self.obs_r_mag_limit))
        self.use_lsst = kwargs.get('use_lsst', False)
        self.exclude_agn = kwargs.get('exclude_agn', False)
        self.plot_shift = kwargs.get('plot_shift', True)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.replace_cat_name = kwargs.get('replace_cat_name', {})
        self.title_in_legend = kwargs.get('title_in_legend', False)
        self.legend_location = kwargs.get('legend_location', 'upper left')
        self.skip_statistic = kwargs.get('skip_statistic', False)
        self.font_size = kwargs.get('font_size', 16)
        self.legend_size = kwargs.get('legend_size', 10)
        self.shorten_cat_name = kwargs.get('shorten_cat_name', True)
        
        # bins of color distribution
        self.bins = np.linspace(-1, 4, 2000)
        self.binsize = self.bins[1] - self.bins[0]

        # Load validation catalog and define catalog-specific properties
        self.sdss_path = os.path.join(self.external_data_dir, 'rongpu', 'SpecPhoto_sdss_mgs_extinction_corrected.fits')
        self.deep2_path = os.path.join(self.external_data_dir, 'rongpu', 'DEEP2_uniq_Terapix_Subaru_trimmed_wights_added.fits')
        
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
        elif self.validation_catalog is not None:
            raise ValueError('Validation catalog not recognized')

        # Magnitude and redshift cut
        if self.validation_catalog is not None:
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
        self.obs_color_dist = {}
        if self.validation_catalog is not None:
            self.obs_color_dist = self.get_color_dist(obscat, obs_translate, weights)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        bands = set(sum((c.split('-') for c in self.colors), []))
        if self.rest_frame:
            possible_names = ('Mag_{}_lsst', 'Mag_{}_sdss', 'Mag_true_{}_lsst_z0', 'Mag_true_{}_sdss_z0')
        else:
            possible_lsst_names = (('mag_{}_noagn_lsst', 'mag_true_{}_noagn_lsst')
                                   if self.exclude_agn else ('mag_{}_cModel', 'mag_{}_lsst', 'mag_true_{}_lsst'))
            possible_non_lsst_names = ('mag_{}_sdss', 'mag_{}_des', 'mag_true_{}_sdss', 'mag_true_{}_des')
            if self.use_lsst:
                print('Selecting lsst magnitudes if available')
                possible_names = possible_lsst_names + possible_non_lsst_names
            else:
                possible_names = possible_non_lsst_names + possible_lsst_names

        labels = {band: catalog_instance.first_available(*(n.format(band) for n in possible_names)) for band in bands}
        labels = {k: v for k, v in labels.items() if v}

        if len(labels) < 2:
            return TestResult(skipped=True, summary='magnitudes in mock catalog do not have at least two needed bands.')
        filters = set((v.split('_')[(-2 if 'z0' in v else -1)] for v in labels.values()))

        if len(filters) > 1:
            return TestResult(skipped=True, summary='magnitudes in mock catalog have mixed filters.')
        filter_this = filters.pop()

        if self.lightcone:
            labels['redshift'] = catalog_instance.first_available('redshift_true_galaxy', 'redshift_true', 'redshift')
            if not labels['redshift']:
                return TestResult(skipped=True, summary='mock catalog does not have redshift.')

            # Load mock catalog data
            filters = ['{} > {}'.format(labels['redshift'], self.zlo),
                       '{} < {}'.format(labels['redshift'], self.zhi)]
        else:
            filters = None
            redshift = catalog_instance.redshift

        data = catalog_instance.get_quantities(list(labels.values()), filters)
        # filter catalog data further for matched object catalogs 
        if np.ma.isMaskedArray(data[labels['redshift']]):
            galmask = np.ma.getmask(data[labels['redshift']])
            data = {k:data[v][galmask] for k, v in labels.items()}
        else:
            data = {k: data[v] for k, v in labels.items()}

        # Color transformation
        color_trans = None
        if self.color_transformation_q:
            color_trans_name = None
            if self.validation_catalog == 'DEEP2' and (filter_this == 'sdss' or filter_this == 'des'):
                color_trans_name = '{}2cfht'.format(filter_this)
            elif self.validation_catalog == 'SDSS' and filter_this == 'des':
                color_trans_name = 'des2sdss'
            if color_trans_name:
                color_trans = color_transformation[color_trans_name]

        filter_title = r'\mathrm{{{}}}'.format(filter_this.upper())
        if color_trans:
            data_transformed = {}
            for band in bands:
                try:
                    data_transformed[band] = ne.evaluate(color_trans[band], local_dict=data, global_dict={})
                except KeyError:
                    continue

            filter_title = (r'{}\rightarrow\mathrm{{{}}}'.format(filter_title, self.validation_catalog)
                            if data_transformed else filter_title)
            data_transformed['redshift'] = data['redshift']
            data = data_transformed
            del data_transformed

        if self.obs_r_mag_limit and not self.rest_frame:
            data = GCRQuery('r < {}'.format(self.obs_r_mag_limit)).filter(data)
        elif self.Mag_r_limit and self.rest_frame:
            data = GCRQuery('r < {}'.format(self.Mag_r_limit)).filter(data)

        # Compute color distribution (PDF, CDF etc.)
        mock_color_dist = self.get_color_dist(data)

        # Calculate Cramer-von Mises statistic
        color_shift = {}
        cvm_omega = {}
        cvm_omega_shift = {}
        if self.validation_catalog:
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

        redshift_title = '{:.2f} < z < {:.2f}'.format(self.zlo,
                                                      self.zhi) if self.lightcone else 'z = {:.2f}'.format(redshift)

        self.make_plots(mock_color_dist, color_shift, cvm_omega, cvm_omega_shift, catalog_name,
                        output_dir, filter_title, redshift_title)

        # Write to summary file
        fn = os.path.join(output_dir, self.summary_output_file)
        with open(fn, 'a') as f:
            if color_trans:
                f.write('Color transformation: {}\n'.format(color_trans_name))
            else:
                f.write('No color transformation\n')
            f.write('{}\n'.format(redshift_title))
            if self.obs_r_mag_limit:
                f.write('r_mag < %2.3f\n\n'%(self.obs_r_mag_limit))
            elif self.Mag_r_limit:
                f.write('Mag_r < %2.3f\n\n'%(self.Mag_r_limit))
            if self.validation_catalog:
                for color in self.colors:
                    if self.validation_catalog and not ((color in self.obs_color_dist) and (color in mock_color_dist)):
                        continue
                    f.write("Median "+color+" difference (obs - mock) = %2.3f\n"%(color_shift[color]))
                    f.write(color+": {} = {:2.6f}\n".format('CvM statistic', cvm_omega[color]))
                    f.write(color+" (shifted): {} = {:2.6f}\n".format('CvM statistic', cvm_omega_shift[color]))
                    f.write("\n")

        return TestResult(inspect_only=True)


    def make_plots(self, mock_color_dist, color_shift, cvm_omega, cvm_omega_shift, catalog_name,
                   output_dir, filter_title, redshift_title):
        available_colors = [c for c in self.colors if c in mock_color_dist]

        nrows = int(np.ceil(len(available_colors)/2.))
        fig_pdf, axes_pdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))
        fig_cdf, axes_cdf = plt.subplots(nrows, 2, figsize=(8, 3.5*nrows))
        title = ''
        if self.obs_r_mag_limit:
            title = '$m_r^{{{}}} < {:2.1f},  {}$'.format(filter_title, self.obs_r_mag_limit, redshift_title)
        elif self.Mag_r_limit:
            title = '$M_r^{{{}}} < {:2.1f},  {}$'.format(filter_title, self.Mag_r_limit, redshift_title)

        for ax_cdf, ax_pdf, color in zip(axes_cdf.flat, axes_pdf.flat, available_colors):

            if color not in mock_color_dist or (self.validation_catalog and color not in self.obs_color_dist):
                continue
            mbinctr = mock_color_dist[color]['binctr']
            mpdf_smooth = mock_color_dist[color]['pdf_smooth']
            mcdf = mock_color_dist[color]['cdf']
            if self.validation_catalog:
                obinctr = self.obs_color_dist[color]['binctr']
                opdf_smooth = self.obs_color_dist[color]['pdf_smooth']
                ocdf = self.obs_color_dist[color]['cdf']
                xmin = np.min([mbinctr[find_first_true(mcdf > 0.001)],
                               mbinctr[find_first_true(mcdf > 0.001)] + color_shift[color],
                               obinctr[find_first_true(ocdf > 0.001)]])
                xmax = np.max([mbinctr[find_first_true(mcdf > 0.999)],
                               mbinctr[find_first_true(mcdf > 0.999)] + color_shift[color],
                               obinctr[find_first_true(ocdf > 0.999)]])
            else:
                xmin = np.min(mbinctr[find_first_true(mcdf > 0.001)])
                xmax = np.max(mbinctr[find_first_true(mcdf > 0.999)])

            # Plot PDF
            # mock color distribution
            spacing = '\n'
            lgnd_title = None
            if self.truncate_cat_name:
                catalog_name = re.split('_', catalog_name)[0]
                spacing = ', '
            if self.replace_cat_name:
                for k, v in self.replace_cat_name.items():
                    catalog_name = re.sub(k, v, catalog_name)
                
            if cvm_omega.get(color, None) and not self.skip_statistic:
                catalog_label = catalog_name + spacing + r'$\omega={:.3}$'.format(cvm_omega[color])
            else:
                catalog_label = catalog_name
            ax_pdf.step(mbinctr, mpdf_smooth, where="mid", label=catalog_label, color='C1')
            if self.validation_catalog:
                # validation data
                ax_pdf.step(obinctr, opdf_smooth, where="mid", label=self.validation_catalog, color='C0')
                # color distribution after constant shift
                if self.plot_shift:
                    ax_pdf.step(mbinctr + color_shift[color], mpdf_smooth,
                                label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift[color]),
                                linestyle='--', color='C2')
            ax_pdf.set_xlabel('${}$'.format(color), size=self.font_size)
            ax_pdf.set_xlim(xmin, xmax)
            ax_pdf.set_ylim(bottom=0.)
            if not self.title_in_legend:
                ax_pdf.set_title(title)
            else:
                lgnd_title = title
            ax_pdf.legend(loc=self.legend_location, title=lgnd_title, fontsize=self.legend_size, frameon=False)

            # Plot CDF
            # catalog distribution
            ax_cdf.step(mbinctr, mcdf, where="mid", label=catalog_label, color='C1')
            if self.validation_catalog:
                # validation distribution
                ax_cdf.step(obinctr, ocdf, label=self.validation_catalog, color='C0')
                # color distribution after constant shift
                if self.plot_shift:
                    ax_cdf.step(mbinctr + color_shift[color], mcdf, where="mid",
                                label=catalog_name+' shifted\n'+r'$\omega={:.3}$'.format(cvm_omega_shift[color]),
                                linestyle='--', color='C2')
            ax_cdf.set_xlabel('${}$'.format(color), size=self.font_size)
            if not self.title_in_legend:
                ax_cdf.set_title(title)
            else:
                lgnd_title = title
            ax_cdf.set_xlim(xmin, xmax)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.legend(loc=self.legend_location, title=lgnd_title, fontsize=self.legend_size, frameon=False)

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
                if self.rest_frame:
                    cat_mask = (cat[band1] < -10) & (cat[band1] > -30) & (cat[band2] < -10) & (cat[band2] > -30)
                else:
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
