from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
from scipy import interpolate
from scipy.stats import binned_statistic
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .plotting import mpl

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
        self.catalogs = kwargs['catalogs']
        self.observation = kwargs['observation']
        self.possible_mag_fields = kwargs['possible_mag_fields']
        self.test_name = kwargs['test_name']
        self.data_label = kwargs['data_label']
        self.z_bins = kwargs['z_bins']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_subplot_row = kwargs['fig_subplot_row']
        self.fig_subplot_col = kwargs['fig_subplot_col']
        self.suptitle = kwargs['suptitle']
        self.xlim = kwargs['xlim']
        self.chisq_max = kwargs['chisq_max']
        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.genfromtxt(validation_filepath)

    @staticmethod
    def ConvertAbsMagLuminosity(AbsM, band):
        '''AbsM: absolute magnitude, band: filter'''
        if isinstance(AbsM, (list, np.ndarray)):
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
            colnames['bulge_to_total_ratio_i'] = catalog_instance.first_available('bulge_to_total_ratio_i')

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

        fig, axes = plt.subplots(self.fig_subplot_row, self.fig_subplot_col, figsize=(self.fig_subplot_col*4, self.fig_subplot_row*4), sharex=True, sharey=True)
        twocomp_labels = [r'$R_B^{B/T > 0.5}$', r'$R_B^{B/T < 0.5}$']
        twocomp_sim_labels = [r'Sims:$R_B^{B/T > 0.5}$', r'Sims:$R_D^{B/T < 0.5}$']
        onecomp_labels = ['Simulation', 'Validation']
        list_of_validation_values = []

        val_colors = plt.cm.jet(np.linspace(0.95, 0.25, 2)) # pylint: disable=no-member
        sim_colors = plt.cm.jet(np.linspace(1.0, 0.2, 2)) # pylint: disable=no-member
        default_L_bin_edges = np.array([9, 9.5, 10, 10.5, 11, 11.5])
        default_L_bins = (default_L_bin_edges[1:] + default_L_bin_edges[:-1]) / 2.

        try:
            col = 0
            row = 0
            ylo = 9999
            yhi = -9999
            for z_bin in self.z_bins:
                if self.fig_subplot_row == 1:
                    ax = axes[col]
                else:
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
                ob = mpl.offsetbox.AnchoredText(self.label_template.format(z_bin['z_min'], z_bin['z_max']), loc=1, frameon=False)

                if self.observation == 'onecomp':
                    num_axes = 1
                    logL_G = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], 'g')
                    size_kpc = catalog_data_this['size'] * self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])
                    binned_size_kpc = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='mean')[0]
                    binned_size_kpc_err = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='std')[0]
                    binned_size_N = binned_statistic(logL_G, size_kpc, bins=default_L_bin_edges, statistic='count')[0]
                    binned_size_kpc_err = binned_size_kpc_err / np.sqrt(binned_size_N)

                    np.savetxt(output_filepath, np.transpose((default_L_bins, binned_size_kpc, binned_size_kpc_err)))

                    validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.25) & (self.validation_data[:,0] > z_mean - 0.25)]

                    ax.semilogy(validation_this[:,1], 10**validation_this[:, 2], label=onecomp_labels[1])
                    ax.fill_between(validation_this[:,1], 10**validation_this[:,3], 10**validation_this[:,4], lw=0, alpha=0.2)
                    ax.errorbar(default_L_bins, binned_size_kpc, binned_size_kpc_err, marker='o', ms=9, ls='', label=onecomp_labels[0])
                    onecomp_labels = [None, None]

                    ax.add_artist(ob)
                    tylo, tyhi = np.percentile(size_kpc, [5, 95])
                    tylo_arr = [ylo, 10**validation_this[:,4].min()]
                    if np.any(tylo_arr == 0):
                        ylo = np.partition(tylo_arr, 1)[1]
                    else:
                        ylo = np.min(tylo_arr)

                    tyhi = np.max([yhi, tyhi, 10**validation_this[:,3].max()])
                    if tyhi > yhi:
                        yhi = tyhi

                    validation = self.compute_chisq(default_L_bins, binned_size_kpc, binned_size_kpc_err,
                                                    validation_this[:,1], 10**validation_this[:,2])
                    list_of_validation_values.append(validation)

                elif self.observation == 'twocomp':
                    axes2 = []
                    num_axes = 2
                    logL_I = self.ConvertAbsMagLuminosity(catalog_data_this['mag'], 'i')
                    arcsec_to_kpc = self._ARCSEC_TO_RADIAN * interpolate.splev(catalog_data_this['z'], spl) / (1 + catalog_data_this['z'])

                    bt_cons = [(catalog_data_this['bulge_to_total_ratio_i'] >= 0.5), (catalog_data_this['bulge_to_total_ratio_i'] < 0.5)]
                    to_write = default_L_bins.copy()
                    divider = make_axes_locatable(ax)
                    ax2 = divider.append_axes("top", size='100%', pad=0)
                    axes2.append(ax2)
                    validation_values = []
                    for bti, si, axi, validation_cols, label, color, sim_label, sim_color in zip(
                            bt_cons,
                            ('size_bulge', 'size_disk'),
                            (ax2, ax),
                            ((1, 2, 3), (6, 9, 10)),
                            twocomp_labels,
                            val_colors,
                            twocomp_sim_labels,
                            sim_colors,
                    ):
                        tsize_kpc = binned_statistic(logL_I[bti], (catalog_data_this[si] * arcsec_to_kpc)[bti], bins=default_L_bin_edges, statistic='mean')[0]
                        tsize_kpc_err = binned_statistic(logL_I[bti], (catalog_data_this[si] * arcsec_to_kpc)[bti], bins=default_L_bin_edges, statistic='std')[0]
                        tsize_N = binned_statistic(logL_I[bti], (catalog_data_this[si] * arcsec_to_kpc)[bti], bins=default_L_bin_edges, statistic='count')[0]
                        tsize_kpc = np.nan_to_num(tsize_kpc)
                        tsize_kpc_err = np.nan_to_num(tsize_kpc_err / np.sqrt(tsize_N))

                        ylo, yhi = 9999., -9999.
                        tylo, tyhi = np.percentile(tsize_kpc-tsize_kpc_err, [5, 95])
                        if tylo < ylo:
                            ylo = tylo
                        if tyhi > yhi:
                            yhi = tyhi

                        to_write = np.column_stack((to_write, tsize_kpc, tsize_kpc_err))
                        axi.errorbar(default_L_bins, tsize_kpc, tsize_kpc_err, marker='o', ls='', label=sim_label, c=sim_color)
                        np.savetxt(output_filepath, to_write)

                        validation_this = self.validation_data[(self.validation_data[:,0] < z_mean + 0.25) & (self.validation_data[:,0] > z_mean - 0.25)]
                        ix, iy, iyerr = validation_cols
                        vali_max = validation_this[:, iy] + validation_this[:, iyerr]
                        vali_min = validation_this[:, iy] - validation_this[:, iyerr]
                        axi.semilogy(validation_this[:, ix], validation_this[:, iy], label=label, color=color)
                        axi.fill_between(validation_this[:, ix], vali_max, vali_min, lw=0, alpha=0.2, facecolor=color)
                        validation_values.append(self.compute_chisq(default_L_bins, tsize_kpc, tsize_kpc_err, validation_this[:, ix], validation_this[:, iy]))

                        vali_min_min = vali_min.min()
                        if vali_min_min > 0 and vali_min_min < ylo:
                            ylo = vali_min_min
                        yhi = max(yhi, vali_max.max())

                    ax2.set_xlim(self.xlim)
                    ax.set_yscale('log', nonposy='clip')
                    ax2.set_yscale('log', nonposy='clip')
                    ax2.xaxis.set_ticklabels([])
                    if col > 0:
                        ax2.yaxis.set_ticklabels([])

                    ax2.tick_params(direction='in', which='both')
                    ax2.legend(loc=3, ncol=2, fontsize=10)
                    ax2.add_artist(ob)

                    list_of_validation_values.append(validation_values)
                del catalog_data_this
                twocomp_labels = [None, None]
                twocomp_sim_labels = [None, None]

                ax.set_xlim(self.xlim)
                ax.tick_params(direction='in', which='both')
                ax.legend(loc=3, ncol=2, fontsize=10)

                col += 1
                if col > 2:
                    col = 0
                    row += 1
                if num_axes == 1:
                    for ax in axes:
                        ax.set_ylim([ylo, yhi])
                if num_axes == 2:
                    for ax, ax2 in zip(axes, axes2):
                        ax.set_ylim([ylo, yhi])
                        ax2.set_ylim([ylo, yhi])

            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axes
            plt.tick_params(labelcolor='none', which='both', top='off', bottom='off', left='off', right='off')
            plt.grid(False)
            plt.xlabel(self.fig_xlabel)
            plt.ylabel(self.fig_ylabel)
            fig.subplots_adjust(hspace=0, wspace=0.2)
            fig.suptitle('{} ({}) vs. {}'.format(catalog_name, self.suptitle, self.data_label), fontsize='medium', y=0.98)
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
