from __future__ import unicode_literals, absolute_import, division
from GCR import GCRQuery
from astropy.io import fits
import numpy as np
import sys
import pickle
import os

from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['RedSequence']

rs_path = '/global/projecta/projectdirs/lsst/groups/CS/descqa/data/redsequence/{}_rs_{}.fits'

class RedSequence(BaseValidationTest):

    def __init__(self, **kwargs): #pylint: disable=W0231

        self.kwargs = kwargs
        self.bands = ['g', 'r', 'i', 'z']
        self.n_bands = len(self.bands)

        self.validation_catalog = kwargs.get('validation_catalog', 'des_y1')
        self.rs_z = fits.open(rs_path.format(self.validation_catalog, 'z'))[0].data[:-1]
        self.rs_mean = fits.open(rs_path.format(self.validation_catalog, 'c'))[0].data[:-1]
        self.rs_sigma = fits.open(rs_path.format(self.validation_catalog, 's'))[0].data[...,:-1]

        self.z_bins = np.linspace(*kwargs.get('z_bins', (0.0, 1.0, 31)))

        self.c_bins = np.linspace(*kwargs.get('c_bins', (-0.5, 2.0, 101)))
        
        self.mass_bins = np.logspace(*kwargs.get('mass_bins', (12.5, 14, 4)))

        self.dz = self.z_bins[1:] - self.z_bins[:-1]
        self.dz = np.hstack([self.dz, np.array([self.dz[-1]])])
        self.dc = self.c_bins[1:] - self.c_bins[:-1]
        self.dm = self.mass_bins[1:] - self.mass_bins[:-1]

        self.n_z_bins = len(self.z_bins)-1
        self.n_c_bins = len(self.c_bins)-1
        self.n_mass_bins = len(self.mass_bins) - 1
    
        self.z_mean = (self.z_bins[1:] + self.z_bins[:-1]) / 2
        self.c_mean = (self.c_bins[1:] + self.c_bins[:-1]) / 2
        self.mass_mean = (self.mass_bins[1:] + self.mass_bins[:-1]) / 2
        
        self.use_redmapper = kwargs.get('use_redmapper', False)

        possible_mag_fields = ('mag_true_{}_lsst',
                               'mag_true_{}_des',
                              'mag_true_{}_sdss',
                              )
        self.possible_mag_fields = [[f.format(band) for f in possible_mag_fields] for band in self.bands]

        self.possible_absmag_fields = ('Mag_true_r_lsst_z0',
                                       'Mag_true_r_lsst_z01'
                                       'Mag_true_r_des_z0',
                                       'Mag_true_r_des_z01',
                                       'Mag_true_r_sdss_z0',
                                       'Mag_true_r_sdss_z01',
                                   )
                               
                               
    def prepare_galaxy_catalog(self, gc):

        quantities_needed = {'redshift_true', 'is_central', 'halo_mass', 'halo_id', 'galaxy_id'}

        if gc.has_quantities(['truth/RHALO', 'truth/R200']):
            gc.add_quantity_modifier('r_host', 'truth/RHALO', overwrite=True)
            
            gc.add_quantity_modifier('r_vir', 'truth/R200', overwrite=True)
            quantities_needed.add('r_host')
            quantities_needed.add('r_vir')

        mag_fields = [gc.first_available(*fields) for fields in self.possible_mag_fields]
        quantities_needed.update(mag_fields)

        absolute_magnitude_field = gc.first_available(*self.possible_absmag_fields)
        quantities_needed.add(absolute_magnitude_field)

        if not gc.has_quantities(quantities_needed):
            return

        return absolute_magnitude_field, mag_fields, quantities_needed


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        prepared = self.prepare_galaxy_catalog(catalog_instance)
        if prepared is None:
            return TestResult(skipped=True)

        if self.use_redmapper:
            try:
                redmapper = GCRCatalogs.load_catalog(catalog_name+'_redmapper')
            except:
                return TestResult(skipped=True)
            
            redmapper = redmapper.get_quantities(['galaxy_id'])
            
        absolute_magnitude_field, mag_fields, quantities_needed = prepared
        bins     = (self.z_bins, self.c_bins, self.mass_bins)
        hist_cen = np.zeros((self.n_z_bins, self.n_c_bins, self.n_mass_bins, self.n_bands-1))
        hist_sat = np.zeros_like(hist_cen)
        hist_mem_cen = np.zeros_like(hist_cen)
        hist_mem_sat = np.zeros_like(hist_cen)
        
        print(absolute_magnitude_field)
        cen_query = GCRQuery('is_central & ({} < -19)'.format(absolute_magnitude_field))
        sat_query = GCRQuery('(~is_central) & ({} < -19)'.format(absolute_magnitude_field))
        
        if 'r_host' in quantities_needed and 'r_vir' in quantities_needed:
            sat_query &= GCRQuery('r_host < r_vir')


        for data in catalog_instance.get_quantities(quantities_needed, return_iterator=True):
            cen_mask = cen_query.mask(data)
            sat_mask = sat_query.mask(data)
            if self.use_redmapper:
                mem_mask = np.in1d(data['galaxy_id'], redmapper['galaxy_id'])

            for i in range(self.n_bands-1):
                color = data[mag_fields[i]] - data[mag_fields[i+1]]

                hdata = np.stack([data['redshift_true'], color, data['halo_mass']]).T            
                hist_cen[:,:,:,i] += np.histogramdd(hdata[cen_mask], bins)[0]
                hist_sat[:,:,:,i] += np.histogramdd(hdata[sat_mask], bins)[0]
                if self.use_redmapper:
                    hist_mem_cen[:,:,:,i] += np.histogramdd(hdata[mem_mask & cen_mask], bins)[0]
                    hist_mem_sat[:,:,:,i] += np.histogramdd(hdata[mem_mask & sat_mask], bins)[0]

        data = cen_mask = sat_mask = mem_mask = None
        
        rs_mean, rs_scat, red_frac_sat, red_frac_cen = self.compute_summary_statistics(hist_sat, hist_cen,
                                                                                    hist_mem_sat, hist_mem_cen)

        red_seq = {'rs_mean':rs_mean,
                   'rs_scat':rs_scat,
                   'red_frac_sat':red_frac_sat,
                   'red_frac_cen':red_frac_cen}
        
        self.make_plot(red_seq, hist_cen, hist_sat, hist_mem_cen, hist_mem_sat, catalog_name, os.path.join(output_dir, 'red_sequence.png'))

        return TestResult(inspect_only=True)
    
    def compute_summary_statistics(self, hist_sat, hist_cen, hist_mem_sat, hist_mem_cen):
        """
        Calculate mean, and scatter of red sequence and red fraction. 
        """

        tot_sat = np.sum(hist_sat, axis=(1,3))
        tot_cen = np.sum(hist_cen, axis=(1,3))
        tot_sat_mem = np.sum(hist_mem_sat, axis=(1,3))
        tot_cen_mem = np.sum(hist_mem_cen, axis=(1,3))
        
        
        if self.use_redmapper:
            rs_mean = np.sum(self.c_mean.reshape(1,-1,1,1) * (hist_mem_sat + hist_mem_cen) * self.dc.reshape(1,-1,1,1), axis=1) / np.sum((hist_mem_sat + hist_mem_cen) * self.dc.reshape(1,-1,1,1), axis=1)
            rs_scat = np.sqrt(np.sum((self.c_mean.reshape(1,-1,1,1) - rs_mean.reshape(-1,1,self.n_mass_bins,self.n_bands-1)) ** 2 * (hist_mem_sat + hist_mem_cen) * self.dc.reshape(1,-1,1,1), axis=1) / np.sum((hist_mem_sat + hist_mem_cen) * self.dc.reshape(1,-1,1,1), axis=1))
            
            red_frac_sat = 1 - np.sum((hist_sat - hist_mem_sat)/tot_sat.reshape(-1,1,1,1), axis=(1,2))
            red_frac_cen = 1 - np.sum((hist_cen - hist_mem_cen)/tot_cen.reshape(-1,1,1,1), axis=(1,2))
        else:
            rs_mean = np.sum(self.c_mean.reshape(1,-1,1,1) * (hist_sat + hist_cen) * self.dc.reshape(1,-1,1,1), axis=1) / np.sum((hist_sat + hist_cen) * self.dc.reshape(1,-1,1,1), axis=1)
            rs_scat = np.sqrt(np.sum((self.c_mean.reshape(1,-1,1,1) - rs_mean.reshape(-1,1,self.n_mass_bins,self.n_bands-1)) ** 2 * (hist_sat + hist_cen) * self.dc.reshape(1,-1,1,1), axis=1) / np.sum((hist_sat + hist_cen) * self.dc.reshape(1,-1,1,1), axis=1))
            
            red_frac_sat = None
            red_frac_cen = None
        
        return rs_mean, rs_scat, red_frac_sat, red_frac_cen

    def make_plot(self, red_seq, hist_cen, hist_sat, hist_mem_cen, hist_mem_sat, name, save_to):
        fig, ax = plt.subplots(2, self.n_bands-1, sharex=True, sharey=True, figsize=(12,10), dpi=100)

        for i in range(self.n_bands-1):
            for j in range(self.n_mass_bins):
                ax[0,i].plot(self.z_mean, red_seq['rs_mean'][:,j,i], label=r'{:.2E} < $M_h$ < {:.2E}'.format(self.mass_bins[j], self.mass_bins[j+1]))

            if not self.use_redmapper:
                ax[0,i].imshow(np.sum(hist_cen[:,:,:,i], axis=2).T[::-1,:], extent=[self.z_bins[0], self.z_bins[-1], self.c_bins[0], self.c_bins[-1]])
            else:
                ax[0,i].imshow((np.sum(hist_mem_cen + hist_mem_sat, axis=2))[:,:,i].T[::-1,:], extent=[self.z_bins[0], self.z_bins[-1], self.c_bins[0], self.c_bins[-1]])
                
            ax[0,i].plot(self.rs_z, self.rs_mean[:,i], label=self.validation_catalog)

            for j in range(self.n_mass_bins):
                ax[1,i].plot(self.z_mean, red_seq['rs_scat'][:,j,i], label=r'{:.2E} < $M_h$ < {:.2E}'.format(self.mass_bins[j], self.mass_bins[j+1]))
            ax[1,i].plot(self.rs_z, self.rs_sigma[i,i,:], label=self.validation_catalog)

            ax[0,i].set_ylabel(r'$\bar{%s-%s}$'% (self.bands[i], self.bands[i+1]))
            ax[1,i].set_ylabel(r'$\sigma(%s-%s)$'% (self.bands[i], self.bands[i+1]))
            ax[1,i].set_xlabel(r'$z$')

        ax[0,i].legend(loc='lower right', frameon=False, fontsize='medium')

        ax = fig.add_subplot(111, frameon=False)
        ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax.grid(False)
        ax.set_title(name)
                            
        fig.tight_layout()
        fig.savefig(save_to)
        plt.close(fig)                
                            

        if self.use_redmapper:
            fig, ax = plt.subplots(1, sharex=True, sharey=True, figsize=(12,10), dpi=100)
            
            ax.plot(self.z_mean, red_seq['red_frac_sat'], label='satellites')
            ax.plot(self.z_mean, red_seq['red_frac_cen'], label='centrals')

            ax.legend(loc='lower right', frameon=False, fontsize='medium')

            ax = fig.add_subplot(111, frameon=False)
            ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
            ax.grid(False)
            ax.set_ylabel(r'f_red')
            ax.set_xlabel(r'$z$')
            ax.set_title(name) 
            save_to = save_to.replace('red_sequence', 'red_fraction')
            fig.savefig(save_to)
            plt.close(fig)                                 
