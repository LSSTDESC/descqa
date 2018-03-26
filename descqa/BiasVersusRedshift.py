from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
import scipy.optimize as op
from GCR import GCRQuery
import treecorr
import pyccl as ccl # pylint: disable=import-error
from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import generate_uniform_random_ra_dec_footprint, get_healpixel_footprint

__all__ = ['BiasValidation']


def neglnlike(b, x, y, yerr):
    return 0.5*np.sum((b**2*x-y)**2/yerr**2) # We ignore the covariance


class BiasValidation(BaseValidationTest):
    """
    Validation test of 2pt correlation function
    """

    def __init__(self, **kwargs): #pylint: disable=W0231
        self.data_label = kwargs['data_label']
        self.z_bins = kwargs['z_bins']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.test_name = kwargs['test_name']

        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)
        self._treecorr_config = {
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        self._treecorr_config['sep_units'] = 'deg'

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        # load catalog data
        colnames = dict()
        colnames['z'] = catalog_instance.first_available('redshift', 'redshift_true')
        colnames['ra'] = catalog_instance.first_available('ra', 'ra_true')
        colnames['dec'] = catalog_instance.first_available('dec', 'dec_true')
        
        if not all(v for v in colnames.values()):
            return TestResult(skipped=True, summary='Missing requested quantities')
        print(z_bin for z_bin in self.z_bins)
        filters = [(np.isfinite, c) for c in colnames.values()]
        filters.extend((
            '{} < {}'.format(colnames['z'], max(z_bin['z_max'] for z_bin in self.z_bins)),
            '{} >= {}'.format(colnames['z'], min(z_bin['z_min'] for z_bin in self.z_bins)),
        ))
        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}
        cosmo = ccl.Cosmology(Omega_c=catalog_instance.cosmology.Om0-catalog_instance.cosmology.Ob0,
                              Omega_b=catalog_instance.cosmology.Ob0,
                              h=catalog_instance.cosmology.h,
                              sigma8=0.8, # For now let's assume a value for 0.8
                              n_s=0.96 #We assume this value for the scalar index
                             )
        # create random
        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size*self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside)

        rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='deg', dec_units='deg')
        del rand_ra, rand_dec
        rr = treecorr.NNCorrelation(**self._treecorr_config)
        rr.process(rand_cat)

        fig, ax = plt.subplots(1,2)
        best_fit_bias = []
        z_mean = []
        try:
            for z_bin, color in zip(self.z_bins, plt.cm.plasma_r(np.linspace(0.1, 1, len(self.z_bins)))): #pylint: disable=E1101
                # filter catalog data for this bin
                filters = [
                    'z < {}'.format(z_bin['z_max']),
                    'z >= {}'.format(z_bin['z_min']),
                ]
                catalog_data_this = GCRQuery(*filters).filter(catalog_data)
                z_mean.append(np.mean(catalog_data_this['z']))
                cat = treecorr.Catalog(
                    ra=catalog_data_this['ra'],
                    dec=catalog_data_this['dec'],
                    ra_units='deg',
                    dec_units='deg')
                # Generate N(z) for this bin
                nz, be = np.histogram(catalog_data_this['z'], range=(0,7), bins=700)
                z_cent = 0.5*(be[1:]+be[:-1])
                # Generate CCL tracer object to compute Cls -> w(theta)
                tracer = ccl.ClTracerNumberCounts(cosmo, has_rsd=False, has_magnification=False,
                                                  bias=np.ones_like(z_cent), z=z_cent, n=nz)
                del catalog_data_this
                 
                treecorr_config = self._treecorr_config.copy()
                dd = treecorr.NNCorrelation(treecorr_config)
                dr = treecorr.NNCorrelation(treecorr_config)
                rd = treecorr.NNCorrelation(treecorr_config)

                dd.process(cat)
                dr.process(rand_cat, cat)
                rd.process(cat, rand_cat)

                output_filepath = os.path.join(output_dir, self.output_filename_template.format(z_bin['z_min'], z_bin['z_max']))
                dd.write(output_filepath, rr, dr, rd)

                xi, var_xi = dd.calculateXi(rr, dr, rd)
                xi_rad = np.exp(dd.meanlogr)
                xi_sig = np.sqrt(var_xi)
                ells = np.arange(0, 5000)
                cls = ccl.angular_cl(cosmo, tracer, tracer, ells)
                w_th = ccl.correlation(cosmo, ells, cls, xi_rad)
                angles = (xi_rad > z_bin['min_theta']) & (xi_rad < z_bin['max_theta']) # This is not really linear bias regime... 
                result = op.minimize(neglnlike, [1.0], args=(w_th[angles], xi[angles], xi_sig[angles]), bounds=[(0.5,10)])
                best_bias = result['x']
                ax[0].errorbar(xi_rad, xi, xi_sig, marker='o', ls='', c=color)
                ax[0].loglog(xi_rad,best_bias**2*w_th, c=color, label=self.label_template.format(z_bin['z_min'], z_bin['z_max']))
                print('Best fit bias = ', best_bias)
                best_fit_bias.append(best_bias)
            ax[0].legend(loc='best')
            ax[0].set_xlabel(self.fig_xlabel)
            ax[0].set_ylim(*self.fig_ylim)
            ax[0].set_ylabel(self.fig_ylabel)
            ax[0].set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')
            ax[1].plot(z_mean,best_fit_bias,'o')
            ax[1].set_xlabel('$z$')
            ax[1].set_ylabel('$b(z)$')
            plt.tight_layout()
        finally:
            fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
            plt.close(fig)

        #TODO: calculate summary statistics
        return TestResult(inspect_only=True)
