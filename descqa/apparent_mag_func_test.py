from __future__ import unicode_literals, absolute_import, division
import os
import re
import numpy as np
from scipy.interpolate import interp1d
from .utils import get_sky_area
from .base import BaseValidationTest, TestResult
from .plotting import plt


possible_observations = {
    'HSC': {
        'filename_template': 'apparent_mag_func/HSC/hsc_{}_n.dat',
        'usecols': (0, 1, 2),
        'colnames': ('mag', 'n(<mag)', 'err', 'data', 'data_err', 'power_law'),
        'skiprows': 1,
        'label': 'HSC extrapolated (desqagen 2018)',
    }
}

__all__ = ['ApparentMagFuncTest']


class ApparentMagFuncTest(BaseValidationTest):
    """
    cumulative apparent magnitude function test
    """
    def __init__(self, band='r', band_lim=(24.0, 27.5), fractional_tol=0.4, observation='HSC', **kwargs):
        """
        parameters
        ----------
        band : string
            photometric band

        band_lim : float
            apparent magnitude lower and upper limits

        fractional_tol : float
            fractional tolerance allowed between mock and apparent mag func for test to pass

        observation : string
            string indicating which obsrvational data to use for validating
        """
        # pylint: disable=super-init-not-called

        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.title_in_legend = kwargs.get('title_in_legend', False)
        self.skip_label_detail = kwargs.get('skip_label_detail', False)
        self.font_size = kwargs.get('font_size', 16)
        self.legend_size = kwargs.get('legend_size', 10)
        self.x_lower_limit = kwargs.get('x_lower_limit', 15)
        self.print_title = kwargs.get('print_title', False)
        self.min_mag = kwargs.get('min_mag', 19.)
        self.replace_cat_name = kwargs.get('replace_cat_name', {})
        
        # catalog quantities needed
        possible_mag_fields = ('mag_{}_cModel',
                               'mag_{}_lsst',
                               'mag_true_{}_lsst',
                               'mag_{}_sdss',
                               'mag_true_{}_sdss',
                               'mag_{}_des',
                               'mag_true_{}_des',
                               'mag_{}_hsc',
                               'mag_true_{}_hsc',
                              )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]

        # attach some attributes to the test
        self.band = band
        self.band_lim = list(band_lim)
        self.fractional_tol = fractional_tol

        # set color of lines in plots
        colors = plt.cm.jet(np.linspace(0, 1, 5)) # pylint: disable=no-member
        if band == 'g': self.line_color = colors[0]
        elif band == 'r': self.line_color = colors[1]
        elif band == 'i': self.line_color = colors[2]
        elif band == 'z': self.line_color = colors[3]
        elif band == 'y': self.line_color = colors[4]
        else: self.line_color = 'black'

        # check for validation observation
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown.')
        elif observation not in possible_observations:
            raise ValueError('Observation: {} not available for this test.'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)

        # prepare summary plot
        self.summary_fig = plt.figure()
        upper_rect = 0.2, 0.4, 0.7, 0.55
        lower_rect = 0.2, 0.125, 0.7, 0.275
        self.summary_upper_ax, self.summary_lower_ax = self.summary_fig.add_axes(upper_rect), self.summary_fig.add_axes(lower_rect)

    def get_validation_data(self, band, observation):
        """
        load (observational) data to use for validation test
        """
        data_args = possible_observations[observation]
        data_path = os.path.join(self.data_dir, data_args['filename_template'].format(band))

        if not os.path.exists(data_path):
            raise ValueError("{}-band data file {} not found".format(band, data_path))

        if not os.path.getsize(data_path):
            raise ValueError("{}-band data file {} is empty".format(band, data_path))

        data = np.loadtxt(data_path, unpack=True, usecols=data_args['usecols'], skiprows=data_args['skiprows'])

        validation_data = dict(zip(data_args['colnames'], data))
        validation_data['label'] = data_args['label'] if not self.skip_label_detail else data_args['label'].rpartition('(')[0]

        return validation_data


    def post_process_plot(self, upper_ax, lower_ax):
        """
        """

        #upper panel
        lgnd_title = ''
        title = str(self.band_lim[0]) + ' < '+self.band + ' < ' + str(self.band_lim[1])
        if self.title_in_legend:
            lgnd_title = title
        elif self.print_title:
            upper_ax.set_title(title)
        upper_ax.legend(loc='upper left', title=lgnd_title, fontsize=self.legend_size)
        upper_ax.set_ylabel(r'$n(< {\rm mag}) ~[{\rm deg^{-2}}]$', size=self.font_size)
        upper_ax.xaxis.set_visible(False)
        upper_ax.set_ylim([1000, 10**7])
        upper_ax.fill_between([self.band_lim[0], self.band_lim[1]], [0, 0], [10**9, 10**9], alpha=0.1, color='grey')
        upper_ax.set_yscale('log')
        upper_ax.set_xlim([self.x_lower_limit, 30])

        #lower panel
        lower_ax.fill_between([self.band_lim[0], self.band_lim[1]], [-1, -1], [1, 1], alpha=0.1, color='grey')
        lower_ax.set_xlabel(self.band + ' magnitude', size=self.font_size)
        lower_ax.set_ylabel(r'$\Delta n/n$', size=self.font_size)
        lower_ax.set_ylim([-1, 1])
        lower_ax.set_yticks([-0.6, 0.0, 0.6])
        lower_ax.set_xlim([self.x_lower_limit, 30])


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        """
        """

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field_key:
            return TestResult(skipped=True, summary='Catalog is missing requested quantity: {}'.format(self.possible_mag_fields))

        # check to see if catalog is a light cone
        # this is required since we must be able to calculate the angular area
        # if attribute `lightcone` does not exist, allow the catalog to proceed
        if not getattr(catalog_instance, 'lightcone', True):
            return TestResult(skipped=True, summary="Catalog is not a light cone.")

        # obtain or calculate sky area
        sky_area = getattr(catalog_instance, 'sky_area', None)
        if sky_area is None:
            if not catalog_instance.has_quantities(['ra', 'dec']):
                return TestResult(skipped=True, summary="'ra' and/or 'dec' not available to compute sky area")
            sky_area = get_sky_area(catalog_instance) # compute area from ra and dec

        sky_area_label = ' (Sky Area = {:.1f} $\\rm deg^2$)'.format(sky_area)

        #####################################################
        # caclulate the cumulative number density of galaxies
        #####################################################

        # filter on extended sources if quantity is available in catalog (eg. in object catalog)
        filters = ['extendedness == 1'] if catalog_instance.has_quantity('extendedness') else None

        # retreive data from mock catalog
        d = catalog_instance.get_quantities([mag_field_key], filters=filters)
        m = d[mag_field_key]
        m = np.sort(m)  # put into order--bright to faint

        # get the total number of galaxies in catalog
        N_tot = len(m)
        N = np.cumsum(np.ones(N_tot))/sky_area

        # define the apparent magnitude bins for plotting purposes
        dmag = 0.1 # bin widths
        max_mag = self.band_lim[1] + 1.0  # go one mag beyond the limit
        min_mag = self.min_mag  # start at bright galaxies
        mag_bins = np.arange(min_mag, max_mag, dmag)

        # calculate N(<mag) at the specified points
        inds = np.searchsorted(m, mag_bins)
        mask = (inds >= len(m))
        inds[mask] = -1 # take care of edge case
        sampled_N = N[inds]

        #################################################
        # plot the cumulative apparent magnitude function
        #################################################

        fig = plt.figure()
        upper_rect = 0.2, 0.4, 0.7, 0.55
        lower_rect = 0.2, 0.125, 0.7, 0.275
        upper_ax, lower_ax = fig.add_axes(upper_rect), fig.add_axes(lower_rect)

        # plot on both this plot and any summary plots
        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]
        if self.replace_cat_name:
            for k, v in self.replace_cat_name.items():      
                catalog_name = re.sub(k, v, catalog_name)
                
        upper_ax.plot(mag_bins, sampled_N, '-', label=catalog_name + sky_area_label)
        self.summary_upper_ax.plot(mag_bins, sampled_N, '-', label=catalog_name + sky_area_label)

        # plot validation data
        n = self.validation_data['n(<mag)']
        m = self.validation_data['mag']
        upper_ax.plot(m, n, '-', label=self.validation_data['label'], color='black')
        upper_ax.fill_between(m, n-self.fractional_tol*n, n+self.fractional_tol*n, color='black', alpha=0.25)

        #################################
        # determine if the catalog passes
        #################################

        # interpolate the validation data in order to compare to the mock catalog at same points
        non_zero_mask = (self.validation_data['n(<mag)'] > 0.0)
        x = self.validation_data['mag'][non_zero_mask]
        y = np.log10(self.validation_data['n(<mag)'])[non_zero_mask]
        f_xy = interp1d(x, y, fill_value='extrapolate')
        nn = 10**f_xy(mag_bins)

        # calculate the fractional diffrence between the mock catalog and validation data
        delta = (sampled_N-nn)/nn

        # find maximum fractional difference in test range
        test_range_mask = (mag_bins >= self.band_lim[0]) & (mag_bins <= self.band_lim[1])
        max_frac_diff = np.max(np.fabs(delta[test_range_mask]))

        # plot on both this plot and any summary plots
        lower_ax.fill_between(m, 0.0*m-self.fractional_tol, 0.0*m+self.fractional_tol, color='black', alpha=0.25)
        lower_ax.plot(m, m*0.0, '-', color='black')
        lower_ax.plot(mag_bins, delta, '-')

        self.summary_lower_ax.plot(mag_bins, delta, '-', label=catalog_name)

        # apply 'passing' criterion
        if max_frac_diff > self.fractional_tol:
            score = max_frac_diff
            passed = False
        else:
            score = max_frac_diff
            passed = True

        self.post_process_plot(upper_ax, lower_ax)
        fig.savefig(os.path.join(output_dir, 'cumulative_app_mag_plot.png'))
        plt.close(fig)

        return TestResult(score, passed=passed)

    def conclude_test(self, output_dir):
        """
        """

        # plot verifaction data on summary plot
        n = self.validation_data['n(<mag)']
        m = self.validation_data['mag']
        self.summary_upper_ax.plot(m, n, '-', label=self.validation_data['label'], color='black')
        self.summary_upper_ax.fill_between(m, n-self.fractional_tol*n, n+self.fractional_tol*n, color='black', alpha=0.25)

        self.summary_lower_ax.fill_between(m, 0.0*m-self.fractional_tol, 0.0*m+self.fractional_tol, color='black', alpha=0.25)
        self.summary_lower_ax.plot(m, m*0.0, '-', color='black')

        self.post_process_plot(self.summary_upper_ax, self.summary_lower_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))

        plt.close(self.summary_fig)
