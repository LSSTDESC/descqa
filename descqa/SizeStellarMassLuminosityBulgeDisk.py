from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
from scipy import interpolate 
from scipy.stats import binned_statistic

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import generate_uniform_random_ra_dec_footprint, get_healpixel_footprint, generate_uniform_random_dist


__all__ = ['SizeStellarMassLuminosityBulgeDisk']


def redshift2dist(cosmology):
    z = np.arange(0, 5.1, 0.5)
    comov_d = cosmology.comoving_distance(z).to('kpc').value
    spl = interpolate.splrep(z, comov_d)   
    return spl


class SizeStellarMassLuminosityBulgeDisk(BaseValidationTest):
    """
    Validation test of 2pt correlation function
    """
    _ARCSEC_TO_RADIAN = np.pi / 180. / 3600.

    def __init__(self, **kwargs):
        self.simulated_catalog = kwargs['simulated_catalog']
        if self.simulated_catalog == 'protodc':
            self.possible_mag_fields = kwargs['possible_appmag_fields']
            self.data_label = kwargs['protodc_data_label']
            self.output_filename_template = kwargs['protodc_output_filename_template']
            self.mag_bin_separation = kwargs['mag_bin_separation']
            self.fig_xlabel = kwargs['protodc_fig_xlabel']
            self.fig_ylabel = kwargs['protodc_fig_ylabel']

            validation_filepath = os.path.join(self.data_dir, kwargs['protodc_data_filename'])
            self.validation_data = np.genfromtxt(validation_filepath)
        elif self.simulated_catalog == 'buzzard':
            self.possible_mag_fields = kwargs['possible_AbsMag_fields']
            self.data_label = kwargs['buzzard_data_label']
            self.output_filename_template = kwargs['buzzard_output_filename_template']
            self.fig_xlabel = kwargs['buzzard_fig_xlabel']
            self.fig_ylabel = kwargs['buzzard_fig_ylabel']

            validation_filepath = os.path.join(self.data_dir, kwargs['buzzard_data_filename'])
            self.validation_data = np.genfromtxt(validation_filepath)
        self.test_name = kwargs['test_name'] 
        self.z_bins = kwargs['z_bins']
        self.label_template = kwargs['label_template']

    @staticmethod
    def ConvertAbsMagLuminosity(AbsM, band):
        '''AbsM: absolute magnitude, band: filter'''
        if type(AbsM) is list or type(AbsM) is np.ndarray:
            AbsM = np.array(AbsM)

        bands = {'U':5.61, 'B':5.48, 'V':4.83, 'R':4.42, 'I':4.08, 
                 'J':3.64, 'H':3.32, 'K':3.28, 'g':5.33, 'r':4.67, 
                 'i':4.48, 'z':4.42, 'F300W':6.09, 'F450W':5.32, 'F555W':4.85, 
                 'F606W':4.66, 'F702W':4.32, 'F814W':4.15, 'CFHT_U':5.57, 
                 'CFHT_B':5.49, 'CFHT_V':4.81, 'CFHT_R':4.44, 'CFHT_I':4.06, 
                 'NIRI_J':3.64, 'NIRI_H':3.33, 'NIRI_K':3.29}

        if band in bands.keys():
            AbsSun = bands[band]
        else:
            raise ValueError('Filter not implemented')

        logL = (AbsSun - AbsM) / 2.5 #unit of sun
        L = 10**logL
        return L, logL

    @staticmethod
    def ConvertLuminosityAppMag(L, dl):
        '''L: luminosity in a band, dl: luminosity distance'''

        AbsM = -2.5 * np.log10(L)
        AppM = AbsM - 5 + 5 * np.log10(dl * 1e6)
        return AbsM, AppM

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        if self.simulated_catalog == 'protodc':
            # load catalog data
            default_mag_bin_edges = np.arange(17, 32, self.mag_bin_separation)
            default_mag_bins = (default_mag_bin_edges[1:] + default_mag_bin_edges[:-1]) / 2.
            colnames = dict()
            colnames['z'] = 'redshift'
            colnames['size_bulge'] = 'size_bulge_true'
            colnames['size_disk'] = 'size_disk_true'
            colnames['mag'] = catalog_instance.first_available(*self.possible_mag_fields)
            if not all(v for v in colnames.values()):
                return TestResult(skipped=True, summary='Missing requested quantities')
            #Check whether the columns are finite or not
            filters = [(np.isfinite, c) for c in colnames.values()]

            #Select objects within maximum and minimum redshift of all the bins
            filters.extend((
                '{} < {}'.format(colnames['z'], max(z_bin['z_max'] for z_bin in self.z_bins)),
                '{} >= {}'.format(colnames['z'], min(z_bin['z_min'] for z_bin in self.z_bins)),
            ))
            catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
            catalog_data = {k: catalog_data[v] for k, v in colnames.items()}
        elif self.simulated_catalog == 'buzzard':
            # load catalog data
            default_L_bin_edges = np.array([9, 9.5, 10, 10.5, 11, 11.5])
            default_L_bins = (default_L_bin_edges[1:] + default_L_bin_edges[:-1]) / 2.
            spl = redshift2dist(catalog_instance.cosmology)
            
            colnames = dict()
            #colnames['z'] = catalog_instance.first_available('redshift', 'redshift_true')
            #colnames['size'] = catalog_instance.first_available('size')
            colnames['z'] = 'redshift'
            colnames['size'] = 'size'
            colnames['mag'] = catalog_instance.first_available(*self.possible_mag_fields)
            if not all(v for v in colnames.values()):
                return TestResult(skipped=True, summary='Missing requested quantities')
            #Check whether the columns are finite or not
            filters = [(np.isfinite, c) for c in colnames.values()]

            #Select objects within maximum and minimum redshift of all the bins
            filters.extend((
                '{} < {}'.format(colnames['z'], max(z_bin['z_max'] for z_bin in self.z_bins)),
                '{} >= {}'.format(colnames['z'], min(z_bin['z_min'] for z_bin in self.z_bins)),
            ))
            catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
            catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        colors = ['r', 'b']
        try:
            col = 0
            for z_bin in self.z_bins:
                ax = axes[col]
                # filter catalog data for this bin
                filters = [
                    'z < {}'.format(z_bin['z_max']),
                    'z >= {}'.format(z_bin['z_min']),
                ]

                z_mean = (z_bin['z_max'] + z_bin['z_min']) / 2.
                output_filepath = os.path.join(output_dir, self.output_filename_template.format(z_bin['z_min'], z_bin['z_max']))
                catalog_data_this = GCRQuery(*filters).filter(catalog_data)
                if len(catalog_data_this['z']) == 0:
                    continue 
                if self.simulated_catalog == 'protodc':
                    binned_bulgesize_arcsec, tmp_1, tmp_2 = binned_statistic(catalog_data_this['mag'], catalog_data_this['size_bulge'], bins=default_mag_bin_edges, statistic='mean') 
                    binned_bulgesize_arcsec_err, tmp_1, tmp_2 = binned_statistic(catalog_data_this['mag'], catalog_data_this['size_bulge'], bins=default_mag_bin_edges, statistic='std') 
                    binned_disksize_arcsec, tmp_1, tmp_2 = binned_statistic(catalog_data_this['mag'], catalog_data_this['size_disk'], bins=default_mag_bin_edges, statistic='mean') 
                    binned_disksize_arcsec_err, tmp_1, tmp_2 = binned_statistic(catalog_data_this['mag'], catalog_data_this['size_disk'], bins=default_mag_bin_edges, statistic='std') 
                    binned_bulgesize_arcsec = np.nan_to_num(binned_bulgesize_arcsec)
                    binned_bulgesize_arcsec_err = np.nan_to_num(binned_bulgesize_arcsec_err)
                    binned_disksize_arcsec = np.nan_to_num(binned_disksize_arcsec)
                    binned_disksize_arcsec_err = np.nan_to_num(binned_disksize_arcsec_err)
                    np.savetxt(output_filepath, np.transpose((default_mag_bins, binned_bulgesize_arcsec, binned_bulgesize_arcsec_err, binned_disksize_arcsec, binned_disksize_arcsec_err)))

                    validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.25) & (self.validation_data[:,0] > z_mean - 0.25)]
                    
                    #ax.set_title(self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                    ax.text(23, 1, self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                    ax.semilogy(validation_this[:,1], validation_this[:, 2], label='Bulge', color=colors[0])
                    ax.fill_between(validation_this[:,1], validation_this[:, 2] + validation_this[:,4], validation_this[:, 2] - validation_this[:,4], lw=0, alpha=0.2, facecolor=colors[0])
                    ax.semilogy(validation_this[:,1] + 0.5, validation_this[:, 3], label='Disk', color=colors[1])
                    ax.fill_between(validation_this[:,1] + 0.5, validation_this[:, 3] + validation_this[:,5], validation_this[:, 3] - validation_this[:,5], lw=0, alpha=0.2, facecolor=colors[1])
 
                    ax.errorbar(default_mag_bins, binned_bulgesize_arcsec, binned_bulgesize_arcsec_err, marker='o', ls='', c=colors[0])
                    ax.errorbar(default_mag_bins+0.5, binned_disksize_arcsec, binned_disksize_arcsec_err, marker='o', ls='', c=colors[1])
                    ax.set_xlim([17, 29])
                    ax.set_ylim([1e-2, 1e1])
                    ax.set_yscale('log', nonposy='clip')
                elif self.simulated_catalog == 'buzzard':
                    size_kpc = catalog_data_this['size'] * self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])
                    L_G, logL_G = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], 'g')
                    binned_size_kpc, tmp_1, tmp_2 = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='mean')
                    binned_size_kpc_err, tmp_1, tmp_2 = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='std')
                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_size_kpc, binned_size_kpc_err)))

                    validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.02) & (self.validation_data[:,0] > z_mean - 0.02)]

                    ax.semilogy(validation_this[:,1], 10**validation_this[:, 2], label=self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                    ax.fill_between(validation_this[:,1], 10**validation_this[:,3], 10**validation_this[:,4], lw=0, alpha=0.2)
                    ax.errorbar(default_L_bins, binned_size_kpc, binned_size_kpc_err, marker='o', ls='')
                    ax.set_yscale('log', nonposy='clip')
                del catalog_data_this
                col += 1
                ax.legend(loc='best')

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', which='both', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(self.fig_xlabel)
            plt.ylabel(self.fig_ylabel)
            plt.ylabel(self.fig_ylabel)
            fig.subplots_adjust(hspace=0, wspace=0)
            fig.suptitle('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium', y=0.93)
        finally:
            fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
            plt.close(fig)
        
        #TODO: calculate summary statistics
        return TestResult(0, passed=True)
