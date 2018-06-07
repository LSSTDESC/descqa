from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
from scipy import interpolate
from scipy.stats import binned_statistic

from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['SizeStellarMassLuminosity']


def redshift2dist(cosmology):
    z = np.arange(0, 5.1, 0.5)
    comov_d = cosmology.comoving_distance(z).to('kpc').value
    spl = interpolate.splrep(z, comov_d)
    return spl


class SizeStellarMassLuminosity(BaseValidationTest):
    """
    Validation test of 2pt correlation function
    """
    _ARCSEC_TO_RADIAN = np.pi / 180. / 3600.

    def __init__(self, **kwargs):
        #pylint: disable=W0231
        self.kwargs = kwargs
        self.observation = kwargs['observation']
        self.possible_mag_fields = kwargs['possible_mag_fields']
        self.test_name = kwargs['test_name']
        self.data_label = kwargs['data_label']
        self.z_bins = kwargs['z_bins']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.chisq_max = kwargs['chisq_max']

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.genfromtxt(validation_filepath)

    @staticmethod
    def ConvertAbsMagLuminosity(AbsM, band):
        '''AbsM: absolute magnitude, band: filter'''
        AbsM = np.asarray(AbsM)

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
        return logL

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        # load catalog data
        spl = redshift2dist(catalog_instance.cosmology)

        colnames = dict()
        colnames['z'] = catalog_instance.first_available('redshift', 'redshift_true')
        colnames['mag'] = catalog_instance.first_available(*self.possible_mag_fields)
        if self.observation == 'onecomp':
            colnames['size'] = catalog_instance.first_available('size', 'size_true')
        elif self.observation == 'twocomp':
            colnames['size_bulge'] = catalog_instance.first_available('size_bulge', 'size_bulge_true')
            colnames['size_disk'] = catalog_instance.first_available('size_disk', 'size_disk_true')

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

        fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)
        list_of_validation_values = []
        try:
            col = 0
            row = 0
            for z_bin in self.z_bins:
                ax = axes[row, col]
                # filter catalog data for this bin
                filters = [
                    'z < {}'.format(z_bin['z_max']),
                    'z >= {}'.format(z_bin['z_min']),
                ]

                catalog_data_this = GCRQuery(*filters).filter(catalog_data)
                if len(catalog_data_this['z']) == 0:
                    continue
                z_mean = (z_bin['z_max'] + z_bin['z_min']) / 2.
                output_filepath = os.path.join(output_dir, self.output_filename_template.format(z_bin['z_min'], z_bin['z_max']))
                colors = ['r', 'b']
                default_L_bin_edges = np.array([9, 9.5, 10, 10.5, 11, 11.5])
                default_L_bins = (default_L_bin_edges[1:] + default_L_bin_edges[:-1]) / 2.
                if self.observation == 'onecomp':
                    logL_G = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], 'g')
                    size_kpc = catalog_data_this['size'] * self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])
                    binned_size_kpc = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_size_kpc_err = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='std')[0]

                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_size_kpc, binned_size_kpc_err)))

                    validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.25) & (self.validation_data[:,0] > z_mean - 0.25)]

                    ax.semilogy(validation_this[:,1], 10**validation_this[:,2], label=self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                    ax.fill_between(validation_this[:,1], 10**validation_this[:,3], 10**validation_this[:,4], lw=0, alpha=0.2)
                    ax.errorbar(default_L_bins, binned_size_kpc, binned_size_kpc_err, marker='o', ls='')
                    
                    validation = self.compute_chisq(default_L_bins, binned_size_kpc, binned_size_kpc_err,
                                                    validation_this[:,1], 10**validation_this[:,2])
                    list_of_validation_values.append(validation)                                
                elif self.observation == 'twocomp':
                    logL_I = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], 'i')
                    arcsec_to_kpc = self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])

                    binned_bulgesize_kpc = binned_statistic(logL_I, catalog_data_this['size_bulge'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_bulgesize_kpc_err = binned_statistic(logL_I, catalog_data_this['size_bulge'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    binned_disksize_kpc = binned_statistic(logL_I, catalog_data_this['size_disk'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_disksize_kpc_err = binned_statistic(logL_I, catalog_data_this['size_disk'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    binned_bulgesize_kpc = np.nan_to_num(binned_bulgesize_kpc)
                    binned_bulgesize_kpc_err = np.nan_to_num(binned_bulgesize_kpc_err)
                    binned_disksize_kpc = np.nan_to_num(binned_disksize_kpc)
                    binned_disksize_kpc_err = np.nan_to_num(binned_disksize_kpc_err)
                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err, binned_disksize_kpc, binned_disksize_kpc_err)))

                    validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.25) & (self.validation_data[:,0] > z_mean - 0.25)]

                    ax.text(11, 0.3, self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                    ax.semilogy(validation_this[:,1], validation_this[:,2], label='Bulge', color=colors[0])
                    ax.fill_between(validation_this[:,1], validation_this[:,2] + validation_this[:,4], validation_this[:,2] - validation_this[:,4], lw=0, alpha=0.2, facecolor=colors[0])
                    ax.semilogy(validation_this[:,1] + 0.2, validation_this[:,3], label='Disk', color=colors[1])
                    ax.fill_between(validation_this[:,1] + 0.2, validation_this[:,3] + validation_this[:,5], validation_this[:,3] - validation_this[:,5], lw=0, alpha=0.2, facecolor=colors[1])

                    ax.errorbar(default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err, marker='o', ls='', c=colors[0])
                    ax.errorbar(default_L_bins+0.2, binned_disksize_kpc, binned_disksize_kpc_err, marker='o', ls='', c=colors[1])
                    ax.set_xlim([9, 13])
                    ax.set_ylim([1e-1, 25])
                    ax.set_yscale('log', nonposy='clip')

                    validation_bulge = self.compute_chisq(default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err,
                                                    validation_this[:,1], validation_this[:,2])
                    validation_disk = self.compute_chisq(default_L_bins, binned_disksize_kpc, binned_disksize_kpc_err,
                                                    validation_this[:,1]+0.2, validation_this[:,3])
                    list_of_validation_values.append([validation_bulge, validation_disk])                                
                del catalog_data_this

                col += 1
                if col > 2:
                    col = 0
                    row += 1

                ax.legend(loc='best')

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', which='both', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(self.fig_xlabel)
            plt.ylabel(self.fig_ylabel)
            fig.subplots_adjust(hspace=0, wspace=0.2)
            fig.suptitle('{} ($M_V$) vs. {}'.format(catalog_name, self.data_label), fontsize='medium', y=0.93)
        finally:
            fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
            plt.close(fig)
        allpass = True
        for validation_val, zbin in zip(list_of_validation_values, self.z_bins):
            if hasattr(validation_val, '__iter__'):
                print("Redshift bin {}-{}: bulge chi-square/dof: {}, disk chi-square/dof: {}.".format(
                        zbin['z_min'], zbin['z_max'], validation_val[0], validation_val[1]))
                if validation_val[0] > self.chisq_max:
                    print("Chi-square/dof with respect to validation data is too large for bulges in redshift bin {}-{}".format(
                            zbin['z_min'], zbin['z_max']))
                    allpass = False
                if validation_val[1] > self.chisq_max:
                    print("Chi-square/dof with respect to validation data is too large for disks in redshift bin {}-{}".format(
                            zbin['z_min'], zbin['z_max']))
                    allpass = False
            else:
                print("Redshift bin {}-{}: chi-square/dof: {}.".format(
                        zbin['z_min'], zbin['z_max'], validation_val))
                if validation_val > self.chisq_max:
                    print("Chi-square/dof with respect to validation data is too large for redshift bin {}-{}".format(
                            zbin['z_min'], zbin['z_max']))
                    allpass = False

        #TODO: calculate summary statistics
        return TestResult(score=np.mean(list_of_validation_values), passed=allpass)
        
    def compute_chisq(self, bins, binned_data, binned_err, validation_points, validation_data):
        if np.any(validation_data==0):
            mask = validation_data!=0
            validation_points = validation_points[mask]
            validation_data = validation_data[mask]
        if validation_points[-1]<validation_points[0]:
            validation_points = validation_points[::-1]
            validation_data = validation_data[::-1]
        validation_at_binpoints = interpolate.CubicSpline(validation_points, validation_data)(bins)
        weights = 1./binned_err**2
        return np.sum(weights*(validation_at_binpoints-binned_data)**2)/len(weights)
                                  
