from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt


possible_observations = {
    'HSC': {
        'filename_template': 'apparent_mag_func/HSC/hsc_{}_n.dat',
        'usecols': (0, 1, 2),
        'colnames': ('mag', 'n(<mag)', 'err'),
        'skiprows': 0,
        'label': 'HSC (D. Campbell, Sprint Week-Dec 2017)',
    }
}

__all__ = ['ApparentMagFuncTest']

class ApparentMagFuncTest(BaseValidationTest):
    """
    apparent magnitude function test
    """
    def __init__(self, band='i', band_lim=27.5, observation='', **kwargs):
        """
        parameters
        ----------
        band : string
            photometric band

        band_lim : float
            apparent magnitude upper limit

        observation : string
            string indicating which obsrvational data to use for validating

        """

        #catalog quantities
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                               'mag_{}_hsc',
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
        pass

    
    @staticmethod
    def get_catalog_data(gc, quantities, filters=None):
        data = {}
        if not gc.has_quantities(quantities):
            return TestResult(skipped=True, summary='Missing requested quantities')

        data = gc.get_quantities(quantities, filters=filters)
        #make sure data entries are all finite
        data = GCRQuery(*((np.isfinite, col) for col in data)).filter(data)

        return data

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field_key:
            return TestResult(skipped=True, summary='Missing requested quantities')
        
        #retreive data
        d = catalog_instance.get_quantities([mag_field_key])
        m = d[mag_field_key]
        m = np.sort(m) #put into order--bright to faint

        #caclulate cumulative number of galaxies less than band_lim
        try:
            sky_area = catalog_instance.sky_area
        except AttributeError:
            print('Warning: this catalog has no sky_area attribute!  Setting sky_area=25 sq deg.')
            sky_area = 25.0
        
        N_tot = len(m)
        N = np.cumsum(np.ones(N_tot))/sky_area
        
        #define magnitude bins for plotting purposes
        self.dmag = 0.1
        self.max_mag = self.band_lim + 1.0
        self.min_mag = 17.7
        mag_bins = np.arange(self.min_mag ,self.max_mag, self.dmag)

        #calculate N at the specified points
        inds = np.searchsorted(m,mag_bins)
        mask = (inds >= len(m))
        inds[mask] = -1
        sampled_N = N[inds]

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            ax_this.plot(mag_bins, sampled_N, '-', label=catalog_name)
            ax_this.plot(self.band_lim, N_tot)
            ax_this.set_yscale('log')
            ax_this.set_ylabel(r'$\rm mag$')
            ax_this.set_ylabel(r'$N(<{\rm mag}){\rm deg}^{-2}$')

        #plot validation data
        for ax_this in (ax, self.summary_ax):
            n = self.validation_data['n']
            m = self.validation_data['mag']
            ax_this.plot(mag_bins, sampled_N, 'o', label=self.validation_data['label'])

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'cumulative_app_mag_plot.png'))
        plt.close(fig)

        score = 0 #calculate your summary statistics
        return TestResult(score, passed=True)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
