from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from scipy.interpolate import interp1d
from .base import BaseValidationTest, TestResult
from .plotting import plt
from scipy.interpolate import interp1d


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

        # catalog quantities needed
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_true_{}_lsst',
                               'mag_{}_sdss',
                               'mag_true_{}_sdss',
                               'mag_{}_des',
                               'mag_true_{}_des',
                               'mag_{}_hsc',
                               'mag_true_{}_hsc')
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]

        # attach some attributes to the test
        self.band = band
        self.band_lim = list(band_lim)
        self.fractional_tol = fractional_tol

        # check for validation observation
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown.')
        elif observation not in possible_observations:
            raise ValueError('Observation: {} not available for this test.'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)
 
        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()

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
        validation_data['label'] = data_args['label']

        return validation_data

    @staticmethod
    def get_catalog_data(gc, quantities, filters=None):
        """
        """
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True, summary='Missing requested quantities')

        data = gc.get_quantities(quantities, filters=filters)

        #make sure all data entries are finite
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data

    def post_process_plot(self, ax):
        """
        """
        ax.legend(loc='upper left')
        ax.set_ylabel(r'$n(< {\rm mag}) ~[{\rm deg^{-2}}]$')
        ax.set_xlabel(self.band + ' magnitude')
        ax.set_ylim([1000, 10**7])
        ax.fill_between([self.band_lim[0], self.band_lim[1]], [0, 0], [10**9, 10**9], alpha=0.1, color='grey')
        ax.set_yscale('log')
        ax.set_title(str(self.band_lim[0]) + ' < '+self.band + ' < ' + str(self.band_lim[1]))

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        """
        """

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field_key:
            return TestResult(skipped=True, summary='Catalog is missing requested quantity: {}'.format(self.possible_mag_fields))

        #####################################################
        # caclulate the cumulative number density of galaxies
        #####################################################

        # retreive data from mock catalog
        d = catalog_instance.get_quantities([mag_field_key])
        m = d[mag_field_key]
        m = np.sort(m)  # put into order--bright to faint

        # check to see if catalog is a light cone
        # this is required since we must be able to calculate the angular area
        if not catalog_instance.lightcone:
            return TestResult(skipped=True, summary="Catalog is not a light cone.")

        # check to see the angular area if an attribute of the catalog
        try:
            sky_area = catalog_instance.sky_area
        except AttributeError:
            return TestResult(skipped=True, summary="Catalog needs an attribute 'sky_area'.")
        
        # get the total number of galaxies in catalog
        N_tot = len(m)
        N = np.cumsum(np.ones(N_tot))/sky_area

        # define the apparent magnitude bins for plotting purposes
        dmag = 0.1 # bin widths
        max_mag = self.band_lim[1] + 1.0  # go one mag beyond the limit
        min_mag = self.band_lim[0] - 1.0  # start at bright galaxies
        mag_bins = np.arange(min_mag, max_mag, dmag)

        # calculate N(<mag) at the specified points
        inds = np.searchsorted(m,mag_bins)
        mask = (inds >= len(m))
        inds[mask] = -1 # take care of edge case
        sampled_N = N[inds]
        
        #################################################
        # plot the cumulative apparent magnitude function
        #################################################

        fig, ax = plt.subplots()

        # plot on both this plot and any summary plots
        for ax_this in (ax, self.summary_ax):
            
            # plot mock catalog data
            ax_this.plot(mag_bins, sampled_N, '-', label=catalog_name)

            ax_this.set_yscale('log')
            ax_this.set_ylabel(r'$\rm mag$')
            ax_this.set_ylabel(r'$N(<{\rm mag})~[{\rm deg}^{-2}]$')
            ax_this.set_xlim([17, 30])
            ax_this.set_ylim([1, 10**8])

        # plot validation data
        for ax_this in [ax]:
            n = self.validation_data['n(<mag)']
            m = self.validation_data['mag']
            ax_this.plot(m, n, '-', label=self.validation_data['label'], color='black')
            ax_this.fill_between(m, n-self.rtol*n, n+self.rtol*n, color='black', alpha=0.5)

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'cumulative_app_mag_plot.png'))
        plt.close(fig)

        #################################
        # determine if the catalog passes
        #################################

        # interpolate the validation data in order to compare to the mock catalog at same points
        non_zero_mask = (self.validation_data['n(<mag)']>0.0)
        x = self.validation_data['mag'][non_zero_mask]
        y = np.log10(self.validation_data['n(<mag)'])[non_zero_mask]
        f_xy = interp1d(x, y, fill_value='extrapolate')
        nn = f_xy(mag_bins)

        # calculate the fractional diffrence between the mock catalog and validation data
        delta = (sampled_N-nn)/nn

        # find maximum fractional difference in test range
        test_range_mask = (mag_bins >= self.min_band_lim) & (mag_bins <= self.max_band_lim)
        max_frac_diff = np.max(delta[test_range_mask])

        # apply 'passing' criterion
        if max_frac_diff>self.fractional_tol:
            score = max_frac_diff
            passed = False
        else:
            score = max_frac_diff
            passed = True

        return TestResult(score, passed=passed)

    def conclude_test(self, output_dir):
        """
        """

        # plot verifaction data on summary plot
        n = self.validation_data['n(<mag)']
        m = self.validation_data['mag']
        self.summary_ax.plot(m, n, '-', label=self.validation_data['label'], color='black')
        self.summary_ax.fill_between(m, n-self.fractional_tol*n, n+self.fractional_tol*n, color='black', alpha=0.5)

        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))

        plt.close(self.summary_fig)
