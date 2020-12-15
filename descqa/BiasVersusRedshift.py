from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
import scipy.optimize as op

from GCR import GCRQuery
import pyccl as ccl

from .base import TestResult
from .CorrelationsTwoPoint import CorrelationsAngularTwoPoint
from .plotting import plt


__all__ = ['BiasValidation']

def neglnlike(b, x, y, yerr):
    return 0.5*np.sum((b**2*x-y)**2/yerr**2) # We ignore the covariance

class BiasValidation(CorrelationsAngularTwoPoint):
    """
    Validation test of 2pt correlation function
    """

    def __init__(self, **kwargs): #pylint: disable=W0231
        super().__init__(**kwargs)
        self.data_label = kwargs['data_label']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.test_name = kwargs['test_name']
        self.fit_range = kwargs['fit_range']
        self.ell_max = kwargs['ell_max'] if 'ell_max' in kwargs.keys() else 20000
        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)
    
    def plot_bias_results(self, corr_data, corr_theory, bias, z, catalog_name, output_dir):
        fig, ax = plt.subplots(1,2)
        colors = plt.cm.plasma_r(np.linspace(0.1, 1, len(self.test_samples))) # pylint: disable=no-member

        for sample_name, color in zip(self.test_samples, colors):
            sample_corr = corr_data[sample_name]
            sample_label = self.test_sample_labels.get(sample_name)
            sample_th = corr_theory[sample_name]
            ax[0].loglog(sample_corr[0], sample_th, c=color, label=sample_label)
            ax[0].errorbar(sample_corr[0], sample_corr[1], sample_corr[2], marker='o', ls='', c=color)
            
        ax[0].legend(loc='best')
        ax[0].set_xlabel(self.fig_xlabel)
        ax[0].set_ylim(*self.fig_ylim)
        ax[0].set_ylabel(self.fig_ylabel)
        ax[0].set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')
        ax[1].plot(z,bias)
        ax[1].set_title('Bias vs redshift', fontsize='medium')
        ax[1].set_xlabel('$z$')
        ax[1].set_ylabel('$b(z)$')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
        plt.close(fig)

    
    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        catalog_data = self.load_catalog_data(catalog_instance=catalog_instance,
                                              requested_columns=self.requested_columns,
                                              test_samples=self.test_samples)

        if not catalog_data:
            return TestResult(skipped=True, summary='Missing requested quantities')

        # Initialize catalog's cosmology
        cosmo = ccl.Cosmology(Omega_c=catalog_instance.cosmology.Om0-catalog_instance.cosmology.Ob0,
                              Omega_b=catalog_instance.cosmology.Ob0,
                              h=catalog_instance.cosmology.h,
                              sigma8=0.8, # For now let's assume a value for 0.8
                              n_s=0.96 #We assume this value for the scalar index
                              )
       
        rand_cat, rr = self.generate_processed_randoms(catalog_data)
        
        correlation_data = dict()
        nz_data = dict()
        correlation_theory = dict()
        best_fit_bias = []
        z_mean = []
        for sample_name, sample_conditions in self.test_samples.items():
            tmp_catalog_data = self.create_test_sample(
                catalog_data, sample_conditions)
            with open(os.path.join(output_dir, 'galaxy_count.dat'), 'a') as f:
                f.write('{} {}\n'.format(sample_name, len(tmp_catalog_data['ra'])))
            if not len(tmp_catalog_data['ra']):
                continue
            z_mean.append(np.mean(tmp_catalog_data['redshift']))
            output_treecorr_filepath = os.path.join(output_dir, 
                self.output_filename_template.format(sample_name))
           
            xi_rad, xi, xi_sig = self.run_treecorr(
                catalog_data=tmp_catalog_data,
                treecorr_rand_cat=rand_cat,
                rr=rr,
                output_file_name=output_treecorr_filepath)
            
            correlation_data[sample_name] = (xi_rad, xi, xi_sig)
            nz, be = np.histogram(tmp_catalog_data['redshift'], range=(0, 2), bins=100)
            zcent = 0.5*(be[1:]+be[:-1])
            nz_data[sample_name] = (zcent, nz*1.0)
            
            # Generate CCL tracer object to compute Cls -> w(theta)
            tracer = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(zcent, nz),
                                              bias=(zcent, np.ones_like(zcent)))

            ells = np.arange(0, self.ell_max) # Reduce ell_max to speed-up
           
            cls = ccl.angular_cl(cosmo, tracer, tracer, ells)
            w_th = ccl.correlation(cosmo, ells, cls, xi_rad)
            angles = (xi_rad > self.fit_range[sample_name]['min_theta']) & \
                     (xi_rad < self.fit_range[sample_name]['max_theta']) # Select the fitting range 
            result = op.minimize(neglnlike, [1.0], 
                                 args=(w_th[angles], xi[angles], 
                                       xi_sig[angles]), bounds=[(0.1, 10)])
            best_bias = result['x']
            correlation_theory[sample_name] = best_bias**2*w_th
            best_fit_bias.append(best_bias)
        z_mean = np.array(z_mean)
        best_fit_bias = np.array(best_fit_bias)
        self.plot_bias_results(corr_data=correlation_data,
                                  catalog_name=catalog_name,
                                  corr_theory=correlation_theory,
                                  bias=best_fit_bias,
                                  z=z_mean,
                                  output_dir=output_dir)

        passed = np.all((best_fit_bias[1:]-best_fit_bias[:-1]) > 0) 
        score = np.count_nonzero((best_fit_bias[:-1]-best_fit_bias[1:])>0)*1.0/(len(best_fit_bias)-1.0)
        return TestResult(score=score, passed=passed,
                  summary="Resulting linear bias obtained from the 2pcf")
