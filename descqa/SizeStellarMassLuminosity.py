from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
import re
from GCR import GCRQuery
from scipy import interpolate
from scipy.stats import binned_statistic
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

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
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.title_in_legend = kwargs.get('title_in_legend', False)
        self.font_size = kwargs.get('font_size', 16)
        self.legend_size = kwargs.get('legend_size', 8)
        self.legend_location = kwargs.get('legend_location', 'best')
        self.survey_label = kwargs.get('survey_label', '')
        self.no_title = kwargs.get('no_title', False)
        self.ncolumns = kwargs.get('ncolumns', 3)
        self.nrows = kwargs.get('nrows', 2)
        self.fig_xlim = kwargs.get('fig_xlim', None)
        self.fig_ylim = kwargs.get('fig_ylim', None)
        if self.fig_xlim is not None:
            self.fig_xlim = [float(x) for x in self.fig_xlim]
        if self.fig_ylim is not None:
            self.fig_ylim = [float(y) for y in self.fig_ylim]

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.genfromtxt(validation_filepath)

        if len(self.survey_label) == 0 and self.no_title:
            self.survey_label = self.data_label
        self.fig_y = 3*self.nrows + 1
        self.fig_x = 3*self.ncolumns + 1
        
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
        filtername = colnames['mag'].split('_')[(-1 if colnames['mag'].startswith('m') else -2)].upper()
        band = colnames['mag'].split('_')[(2 if 'true' in colnames['mag'] else 1)]
        filter_id = '{} {}'.format(filtername, band) if filtername != band else band

        if not all(v for v in colnames.values()):
            return TestResult(skipped=True, summary='Missing requested quantities')
        #Check whether the columns are finite or not
        filters = [(np.isfinite, c) for c in colnames.values()]

        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]

        #Select objects within maximum and minimum redshift of all the bins
        filters.extend((
            '{} < {}'.format(colnames['z'], max(z_bin['z_max'] for z_bin in self.z_bins)),
            '{} >= {}'.format(colnames['z'], min(z_bin['z_min'] for z_bin in self.z_bins)),
        ))
        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        fig, axes = plt.subplots(self.nrows, self.ncolumns, figsize=(self.fig_x, self.fig_y), sharex=True, sharey=True)
        list_of_validation_values = []
        try:
            for z_bin, ax in zip_longest(self.z_bins, axes.flat):
                # filter catalog data for this bin
                if z_bin is None:
                    ax.set_visible(False)
                    continue
                filters = [
                    'z < {}'.format(z_bin['z_max']),
                    'z >= {}'.format(z_bin['z_min']),
                ]
                z_mean = (z_bin['z_max'] + z_bin['z_min']) / 2.
                z_width = (z_bin['z_max'] - z_bin['z_min']) / 2.
                legend_title = self.label_template.format(z_bin['z_min'], z_bin['z_max'])

                catalog_data_this = GCRQuery(*filters).filter(catalog_data)
                maskz = (self.validation_data[:,0] < z_mean + z_width) & (self.validation_data[:,0] > z_mean - z_width)
                maskL = (self.validation_data[:,1] > 0.)
                validation_this = self.validation_data[(maskz) & (maskL)]
                if len(catalog_data_this['z']) == 0 or len(validation_this) == 0:
                    ax.set_visible(False)
                    continue
                output_filepath = os.path.join(output_dir, self.output_filename_template.format(catalog_name, z_bin['z_min'], z_bin['z_max']))
                colors = ['r', 'b']
                default_L_bin_edges = np.array([9, 9.5, 10, 10.5, 11, 11.5])
                default_L_bins = (default_L_bin_edges[1:] + default_L_bin_edges[:-1]) / 2.
                if self.observation == 'onecomp':
                    logL_G = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], band)
                    size_kpc = catalog_data_this['size'] * self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])
                    binned_size_kpc = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_size_kpc_err = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    heading = 'Luminosity Size (kpc), Size Error (kpc)'
                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_size_kpc, binned_size_kpc_err)), fmt='%11.4e', header=heading)

                    ax.semilogy(validation_this[:,1], 10**validation_this[:,2], label=self.survey_label)
                    ax.fill_between(validation_this[:,1], 10**validation_this[:,3], 10**validation_this[:,4], lw=0, alpha=0.2)
                    ax.errorbar(default_L_bins, binned_size_kpc, binned_size_kpc_err, marker='o', ls='', label=' '.join([catalog_name, filter_id]))
                    
                    validation = self.compute_chisq(default_L_bins, binned_size_kpc, binned_size_kpc_err,
                                                    validation_this[:,1], 10**validation_this[:,2])
                    list_of_validation_values.append(validation)                                
                elif self.observation == 'twocomp':
                    logL_I = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], band)
                    arcsec_to_kpc = self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])

                    binned_bulgesize_kpc = binned_statistic(logL_I, catalog_data_this['size_bulge'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_bulgesize_kpc_err = binned_statistic(logL_I, catalog_data_this['size_bulge'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    binned_disksize_kpc = binned_statistic(logL_I, catalog_data_this['size_disk'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_disksize_kpc_err = binned_statistic(logL_I, catalog_data_this['size_disk'] * arcsec_to_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    binned_bulgesize_kpc = np.nan_to_num(binned_bulgesize_kpc)
                    binned_bulgesize_kpc_err = np.nan_to_num(binned_bulgesize_kpc_err)
                    binned_disksize_kpc = np.nan_to_num(binned_disksize_kpc)
                    binned_disksize_kpc_err = np.nan_to_num(binned_disksize_kpc_err)
                    heading = 'Luminosity Bulge-Size (kpc), Bulge Error (kpc) Disk-Size (kpc), Disk Error (kpc)'
                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err, 
                                                              binned_disksize_kpc, binned_disksize_kpc_err)), fmt='%11.4e', header=heading)
                    
                    ax.semilogy(validation_this[:,1], validation_this[:,2], label=' '.join([self.survey_label, 'Bulge']), color=colors[0])
                    ax.fill_between(validation_this[:,1], validation_this[:,2] + validation_this[:,4],
                                    validation_this[:,2] - validation_this[:,4], lw=0, alpha=0.2, facecolor=colors[0])
                    ax.semilogy(validation_this[:,1] + 0.2, validation_this[:,3], label=' '.join([self.survey_label, 'Disk']), color=colors[1])
                    ax.fill_between(validation_this[:,1] + 0.2, validation_this[:,3] + validation_this[:,5], 
                                    validation_this[:,3] - validation_this[:,5], lw=0, alpha=0.2, facecolor=colors[1])

                    ax.errorbar(default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err, marker='o', ls='',
                                c=colors[0], label=' '.join([catalog_name, filter_id]))
                    ax.errorbar(default_L_bins+0.2, binned_disksize_kpc, binned_disksize_kpc_err, marker='o', ls='',
                                c=colors[1], label=' '.join([catalog_name, filter_id]))
                    ax.set_yscale('log', nonposy='clip')

                    validation_bulge = self.compute_chisq(default_L_bins, binned_bulgesize_kpc, binned_bulgesize_kpc_err,
                                                    validation_this[:,1], validation_this[:,2])
                    validation_disk = self.compute_chisq(default_L_bins, binned_disksize_kpc, binned_disksize_kpc_err,
                                                    validation_this[:,1]+0.2, validation_this[:,3])
                    list_of_validation_values.append([validation_bulge, validation_disk])                                
                del catalog_data_this
                if self.fig_xlim is not None:
                    ax.set_xlim(self.fig_xlim)
                if self.fig_xlim is not None:
                    ax.set_ylim(self.fig_ylim)
                ax.legend(loc=self.legend_location, title=legend_title, fontsize=self.legend_size)

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', which='both', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(self.fig_xlabel, size=self.font_size)
            plt.ylabel(self.fig_ylabel, size=self.font_size)
            fig.subplots_adjust(hspace=0, wspace=0)
            if not self.no_title:
                fig.suptitle('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium', y=0.93)
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
        if len(validation_points)>0 and validation_points[-1]<validation_points[0]:
            validation_points = validation_points[::-1]
            validation_data = validation_data[::-1]
        if len(validation_points)>1:
            validation_at_binpoints = interpolate.CubicSpline(validation_points, validation_data)(bins)
        else:
            validation_at_binpoints = binned_data #force chi-sq to zero if no validation data

        weights = 1./binned_err**2
        return np.sum(weights*(validation_at_binpoints-binned_data)**2)/len(weights)
                                  
