from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['ApparentMagFuncTest']

class ApparentMagFuncTest(BaseValidationTest):
    """
    apparent magnitude function test
    """
    def __init__(self, band='i', band_lim=27, observation=None, **kwargs):
        """
        parameters
        ----------
        band : string
            photometric band

        band_lim : float
            apparent magnitude upper magnitude limit
        """

        #catalog quantities
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                              )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]
        self.band = band
        self.band_lim = band_lim

        #check for validation observations
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        elif observation not in possible_observations:
            raise ValueError('Observation {} not available'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()


    def post_process_plot(self, ax):
        pass


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped=True, summary='Missing requested quantities')
        
        #retreive data
        d = gc.get_quantities([mag_field_key])
        m = d[mag_field_key]
        m = np.sort(m) #put into order--bright to faint

        #caclulate cumulative number of galaxies less than band_lim
        mask = (m < self.band_lim)
        N_tot = np.sum(mask)
        N = np.cumsum(np.ones(N_tot))
        
        #define magnitude bins for plotting purposes
        self.dmag = 0.1
        self.max_mag = mag_lim + 1.0
        self.min_mag = 17.7
        mag_bins = np.arange(self.min_mag ,self.max_mag, self.dmag)

        #calculate N at the specified points
        inds = np.searchsorted(m,mag_bins)
        sampled_N = N[inds]

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            ax_this.plot(mag_bins, sampled_N, label=catalog_name)
            ax_this.plot(self.band_lim, N_tot)

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'cumulative_app_mag_plot.png'))
        plt.close(fig)

        score = data[0] #calculate your summary statistics
        return TestResult(score, passed=True)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
