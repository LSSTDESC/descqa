from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from scipy.interpolate import interp1d
from .base import BaseValidationTest, TestResult
from .plotting import plt


possible_observations = {
    'HSC': {
        'filename_template': 'apparent_mag_func/HSC/hsc_{}_n.dat',
        'usecols': (0, 1, 2),
        'colnames': ('mag', 'n(<mag)', 'err'),
        'skiprows': 1,
        'label': 'HSC extrapolated (desqagen 2018)',
    }
}

__all__ = ['ApparentMagFuncTest']


class ApparentMagFuncTest(BaseValidationTest):
    """
    apparent magnitude function test
    """
    def __init__(self, band='i', band_lim=[24, 27.5], rtol=0.2, observation='HSC', **kwargs):
        """
        parameters
        ----------
        band : string
            photometric band

        band_lim : float
            apparent magnitude lower and upper limits

        rtol : float
            relative tolerance alowed between mock and test data

        observation : string
            string indicating which obsrvational data to use for validating

        """

        # catalog quantities needed
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                               'mag_{}_hsc')
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]

        # attach some attributes to the test
        self.band = band
        self.band_lim = band_lim
        self.rtol = rtol

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

    def post_process_plot(self, ax):

        ax.legend(loc='upper left')
        ax.set_ylabel(r'$n(< {\rm mag}) ~[{\rm deg^{-2}}]$')
        ax.set_xlabel(self.band + ' magnitude')
        ax.set_ylim([1000,10**7])
        ax.set_title(str(self.band_lim[0]) + ' < '+self.band + ' < ' + str(self.band_lim[1]))

    @staticmethod
    def get_catalog_data(gc, quantities, filters=None):
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True, summary='Missing requested quantities')

        data = gc.get_quantities(quantities, filters=filters)
        # make sure data entries are all finite
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field_key:
            return TestResult(skipped=True, summary='Catalog is missing requested quantity: {}'.format(self.possible_mag_fields))

        # retreive data from mock catalog
        d = catalog_instance.get_quantities([mag_field_key])
        m = d[mag_field_key]
        m = np.sort(m)  # put into order--bright to faint

        # caclulate cumulative number of galaxies less than band_lim
        if not catalog_instance.lightcone:
            return TestResult(skipped=True, summary="Catalog is not a light cone.")

        try:
            sky_area = catalog_instance.sky_area
        except AttributeError:
            return TestResult(skipped=True, summary="Catalog needs an attribute 'sky_area'.")

        # get total number of galaxies
        N_tot = len(m)
        N = np.cumsum(np.ones(N_tot))/sky_area

        # define apparent magnitude bins for plotting purposes
        self.dmag = 0.1
        self.max_mag = self.band_lim[1] + 1.0  # go one mag beyond the limit
        self.min_mag = self.band_lim[0] - 1.0  # start at bright galaxies
        mag_bins = np.arange(self.min_mag, self.max_mag, self.dmag)

        # calculate N at the specified points
        inds = np.searchsorted(m, mag_bins)
        mask = (inds >= len(m))
        inds[mask] = -1
        sampled_N = N[inds]

        # plot cumulative apparent magnitude function
        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):  # plot on both this and summary plots
            ax_this.plot(mag_bins, sampled_N, '-', label=catalog_name)
            #ax_this.plot(self.band_lim, N_tot)
            ax_this.set_yscale('log')
            ax_this.set_ylabel(r'$\rm mag$')
            ax_this.set_ylabel(r'$N(<{\rm mag})~[{\rm deg}^{-2}]$')
            ax_this.set_xlim([17,30])
            ax_this.set_ylim([1,10**8])

        # plot validation data
        for ax_this in [ax]:
            n = self.validation_data['n(<mag)']
            m = self.validation_data['mag']
            ax_this.plot(m, n, '-', label=self.validation_data['label'], color='black')
            ax_this.fill_between(m, n-self.rtol*n, n+self.rtol*n, color='black', alpha=0.5)

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'cumulative_app_mag_plot.png'))
        plt.close(fig)

        # interpolate upper and lower bounds of test data
        y0 = interp1d(m, np.log10(n))
        y_max = interp1d(m, np.log10(n+self.rtol*n))
        y_min = interp1d(m, np.log10(n-self.rtol*n))
        # interpolate mock data
        y = interp1d(mag_bins, np.log10(sampled_N))

        m_sample = np.linspace(self.band_lim[0], self.band_lim[0], 10000)
        passed = (np.all(y(m_sample) <= y_max(m_sample))) & (np.all(y(m_sample) >= y_min(m_sample)))

        score = np.max(np.fabs((y(m_sample) - y0(m_sample))/y0(m_sample)))

        return TestResult(score, passed=passed)

    def conclude_test(self, output_dir):

        # plot verifaction data on summary plot
        n = self.validation_data['n(<mag)']
        m = self.validation_data['mag']
        self.summary_ax.plot(m, n, '-', label=self.validation_data['label'], color='black')
        self.summary_ax.fill_between(m, n-self.rtol*n, n+self.rtol*n, color='black', alpha=0.5)

        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
