from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['ConditionalLuminosityFunction']

class ConditionalLuminosityFunction(BaseValidationTest):

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.band1 = kwargs.get('band1', 'g')
        self.band2 = kwargs.get('band2', 'r')
        self.magnitude_bins = np.linspace(*kwargs.get('magnitude_bins', (-26, -18, 29)))
        self.mass_bins = np.logspace(*kwargs.get('mass_bins', (13.5, 15, 5)))
        self.z_bins = np.linspace(*kwargs.get('z_bins', (0.2, 1.0, 4)))
        self.color_cut_fraction = float(kwargs.get('color_cut_fraction', 0.2))

        possible_Mag_fields = ('Mag_true_{}_lsst_z0',
                               'Mag_true_{}_lsst_z01',
                               'Mag_true_{}_des_z0',
                               'Mag_true_{}_des_z01',
                               'Mag_true_{}_sdss_z0',
                               'Mag_true_{}_sdss_z01',
                              )

        self.possible_Mag1_fields = [f.format(self.band1) for f in possible_Mag_fields]
        self.possible_Mag2_fields = [f.format(self.band2) for f in possible_Mag_fields]

        color_cut_percentile_at = 100.0 * (1 - self.color_cut_fraction)
        self.color_cut = lambda g, r, z: g-r>np.percentile((g-r)[z<0.2], color_cut_percentile_at)

        self.n_magnitude_bins = len(self.magnitude_bins) - 1
        self.n_mass_bins      = len(self.mass_bins) - 1
        self.n_z_bins         = len(self.z_bins) - 1

        self.dmag = self.magnitude_bins[1:] - self.magnitude_bins[:-1]
        self.mag_center = (self.magnitude_bins[1:] + self.magnitude_bins[:-1])*0.5


    def prepare_galaxy_catalog(self, gc):

        quantities_needed = {'redshift_true', 'is_central', 'halo_mass'}

        if gc.has_quantities(['truth/RHALO', 'truth/R200']):
            gc.add_quantity_modifier('r_host', 'truth/RHALO', overwrite=True)
            gc.add_quantity_modifier('r_vir', 'truth/R200', overwrite=True)
            quantities_needed.add('r_host')
            quantities_needed.add('r_vir')

        absolute_magnitude1_field = gc.first_available(*self.possible_Mag1_fields)
        absolute_magnitude2_field = gc.first_available(*self.possible_Mag2_fields)

        quantities_needed.add(absolute_magnitude1_field)
        quantities_needed.add(absolute_magnitude2_field)

        if not (absolute_magnitude1_field and absolute_magnitude2_field and gc.has_quantities(quantities_needed)):
            return

        return absolute_magnitude1_field, absolute_magnitude2_field, quantities_needed


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        prepared = self.prepare_galaxy_catalog(catalog_instance)
        if prepared is None:
            return TestResult(skipped=True)

        absolute_magnitude1_field, absolute_magnitude2_field, quantities_needed = prepared
        colnames = [absolute_magnitude2_field, 'halo_mass', 'redshift_true']
        bins = (self.magnitude_bins, self.mass_bins, self.z_bins)
        hist_cen = np.zeros((self.n_magnitude_bins, self.n_mass_bins, self.n_z_bins))
        hist_sat = np.zeros_like(hist_cen)

        red_query = GCRQuery((self.color_cut,
                              absolute_magnitude1_field,
                              absolute_magnitude2_field, 'redshift_true'))

        cen_query = GCRQuery('is_central') & red_query
        sat_query = ~GCRQuery('is_central') & red_query

        if 'r_host' in quantities_needed and 'r_vir' in quantities_needed:
            sat_query &= GCRQuery('r_host < r_vir')

        for data in catalog_instance.get_quantities(quantities_needed, return_iterator=True):
            cen_mask = cen_query.mask(data)
            sat_mask = sat_query.mask(data)

            data = np.stack((data[k] for k in colnames)).T
            hist_cen += np.histogramdd(data[cen_mask], bins)[0]
            hist_sat += np.histogramdd(data[sat_mask], bins)[0]

        del data, cen_mask, sat_mask

        halo_counts = hist_cen.sum(axis=0)
        clf = dict()
        clf['sat'] = hist_sat / halo_counts
        clf['cen'] = hist_cen / halo_counts
        clf['tot'] = clf['sat'] + clf['cen']

        self.make_plot(clf, catalog_name, os.path.join(output_dir, 'clf.png'))

        return TestResult(passed=True, score=0)

    def make_plot(self, clf, name, save_to):
        fig, ax = plt.subplots(self.n_mass_bins, self.n_z_bins, sharex=True, sharey=True, figsize=(12,10), dpi=100)

        for i in range(self.n_z_bins):
            for j in range(self.n_mass_bins):
                ax_this = ax[j,i]
                for k, ls in zip(('total', 'satellites', 'centrals'), ('-', ':', '--')):
                    ax_this.semilogy(self.mag_center, clf[k[:3]][:,j,i]/self.dmag, label=k, ls=ls)
                ax_this.set_ylim(0.05, 50)
                bins = self.mass_bins[j], self.mass_bins[j+1], self.z_bins[i], self.z_bins[i+1]
                ax_this.text(-25, 10, '${:.1E}\\leq M <{:.1E}$\n${:g}\\leq z<{:g}$'.format(*bins))

        ax_this.legend(loc='lower right', frameon=False, fontsize='medium')

        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(False)
        ax.set_ylabel(r'$\phi(M_{{{}}}\,|\,M_{{\rm vir}},z)\quad[{{\rm Mag}}^{{-1}}]$'.format(self.band2))
        ax.set_xlabel(r'$M_{{{}}}\quad[{{\rm Mag}}]$'.format(self.band2))
        ax.set_title(name)

        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)
