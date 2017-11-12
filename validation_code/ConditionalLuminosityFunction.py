from __future__ import unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
from .ValidationTest import BaseValidationTest, TestResult, plt

__all__ = ['ConditionalLuminosityFunction']

class ConditionalLuminosityFunction(BaseValidationTest):

    def __init__(self, band='r', magnitude_bins=None, mass_bins=None, z_bins=None, **kwargs):

        possible_Mag_fields = ('Mag_true_{}_lsst_z0',
                               'Mag_true_{}_lsst_z01',
                               'Mag_true_{}_des_z0',
                               'Mag_true_{}_des_z01',
                               'Mag_true_{}_sdss_z0',
                               'Mag_true_{}_sdss_z01',
                              )

        self.possible_Mag_fields = [f.format(band) for f in possible_Mag_fields]
        self.band = band

        self.magnitude_bins   = magnitude_bins or np.linspace(-25, -18, 29)
        self.mass_bins        = mass_bins or np.logspace(12, 15, 5)
        self.z_bins           = z_bins or np.linspace(0, 0.5, 3)

        self.n_magnitude_bins = len(self.magnitude_bins) - 1
        self.n_mass_bins      = len(self.mass_bins) - 1
        self.n_z_bins         = len(self.z_bins) - 1

        self.dmag = self.magnitude_bins[1:] - self.magnitude_bins[:-1]
        self.mag_center = (self.magnitude_bins[1:] + self.magnitude_bins[:-1])*0.5

        self._other_kwargs = kwargs


    def prepare_galaxy_catalog(self, gc):

        quantities_needed = {'redshift_true', 'is_central', 'halo_mass'}

        if gc.has_quantities(['truth/RHALO', 'truth/R200']):
            gc.add_quantity_modifier('r_host', 'truth/RHALO', overwrite=True)
            gc.add_quantity_modifier('r_vir', 'truth/R200', overwrite=True)
            quantities_needed.add('r_host')
            quantities_needed.add('r_vir')

        absolute_magnitude_field = gc.first_available(*self.possible_Mag_fields)
        quantities_needed.add(absolute_magnitude_field)

        return absolute_magnitude_field, gc.get_quantities(quantities_needed)


    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir=None):

        try:
            absolute_magnitude_field, data = self.prepare_galaxy_catalog(galaxy_catalog)
        except (ValueError):
            TestResult(skipped=True)

            
        colnames = [absolute_magnitude_field, 'halo_mass', 'redshift_true']

        cen_query = GCRQuery('is_central')
        sat_query = ~cen_query
        if 'r_host' in data and 'r_vir' in data:
            sat_query &= GCRQuery('r_host < r_vir')

        cen_mask = cen_query.mask(data)
        sat_mask = sat_query.mask(data)

        bins = (self.magnitude_bins, self.mass_bins, self.z_bins)
        data = np.stack((data[k] for k in colnames)).T
        hist_cen = np.histogramdd(data[cen_mask], bins)[0]
        hist_sat = np.histogramdd(data[sat_mask], bins)[0]
        del data, cen_mask, sat_mask

        halo_counts = hist_cen.sum(axis=0)
        clf = dict()
        clf['sat'] = hist_sat / halo_counts
        clf['cen'] = hist_cen / halo_counts
        clf['tot'] = clf['sat'] + clf['cen']

        self.make_plot(clf, catalog_name, os.path.join(base_output_dir, 'clf.png'))

        return TestResult(passed=True, score=0)


    def make_plot(self, clf, name, save_to):

        fig, ax = plt.subplots(self.n_mass_bins, self.n_z_bins, sharex=True, sharey=True, figsize=(12,10), dpi=100)

        for i in range(self.n_z_bins):
            for j in range(self.n_mass_bins):
                ax_this = ax[j,i]
                ax_this.semilogy(self.mag_center, clf['tot'][:,j,i]/self.dmag, label='Total')
                ax_this.semilogy(self.mag_center, clf['sat'][:,j,i]/self.dmag, label='Satellites', linestyle=':')
                ax_this.semilogy(self.mag_center, clf['cen'][:,j,i]/self.dmag, label='Centrals', linestyle='--')
                ax_this.set_ylim(0.05, 50)
                ax_this.text(-25, 10, '${:.1E}\\leq M <{:.1E}$\n${:g}\\leq z<{:g}$'.format(self.mass_bins[j],
                                                                                         self.mass_bins[j+1],
                                                                                         self.z_bins[i],
                                                                                         self.z_bins[i+1]))

        ax_this.legend(loc='lower right', frameon=False, fontsize='medium')

        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(False)
        ax.set_ylabel(r'$\phi(M_{{{}}}\,|\,M_{{\rm vir}},z)\quad[{{\rm Mag}}^{{-1}}]$'.format(self.band))
        ax.set_xlabel(r'$M_{{{}}}\quad[{{\rm Mag}}]$'.format(self.band))
        ax.set_title(name)

        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)

