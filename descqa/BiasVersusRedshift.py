from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
import scipy.optimize as op
import re


from GCR import GCRQuery
import pyccl as ccl

from .base import TestResult
from .CorrelationsTwoPoint import CorrelationsAngularTwoPoint
from .plotting import plt
from .stats import chisq

__all__ = ['BiasValidation']



def neglnlike(b, x, y, yerr):
    return 0.5*np.sum((b**2*x-y)**2/yerr**2) # We ignore the covariance

def wtheta(x, b):
    return b**2*x

class BiasValidation(CorrelationsAngularTwoPoint):
    """
    Validation test of 2pt correlation function
    """

    possible_observations = {'SRD':{
                                'filename_template': 'galaxy_bias/bias_SRD.txt',
                                'label': 'SRD ($i<25.3$)',
                                'colnames': ('z', 'bias'),
                                'skip':2,
                               },
                   'nicola_27':{
                                'filename_template': 'galaxy_bias/bias_nicola_mlim27.txt',
                                'label': 'Nicola et al. \n($i<27$)',
                                'colnames': ('z', 'bias'),
                                'skip':2,
                               },
                 'nicola_25.3':{
                              	'filename_template': 'galaxy_bias/bias_nicola_mlim25.3.txt',
                      	      	'label': 'Nicola et al. \n($i<25.3$)',
                                'colnames': ('z', 'bias'),
                                'skip':2,
                      	       },
                 'nicola_25.3_errors':{
                     'filename_template': 'galaxy_bias/bias_nicola_mlim25.3_with_errors.txt',
                     'label': 'Nicola et al. \n($i<25.3$)',
                     'colnames': ('z', 'b_lo', 'bias', 'b_hi'),
                     'skip':1,
                                      },
                           }

    
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
        self.font_size = kwargs.get('font_size', 16)
        self.legend_size = kwargs.get('legend_size', 12)
        self.title_fontsize = kwargs.get('title_fontsize', 14)
        self.ell_max = kwargs['ell_max'] if 'ell_max' in kwargs.keys() else 20000
        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.title_in_legend = kwargs.get('title_in_legend', True)
        self.observations = kwargs.get('observations', [])

        self.validation_data = self.get_validation_data(self.observations)

    def get_validation_data(self, observations):
    
        validation_data = {}
        if observations:
            for obs in observations:
                print(obs)
                data_args = self.possible_observations[obs]
                fn = os.path.join(self.data_dir, data_args['filename_template'])
                validation_data[obs] = dict(zip(data_args['colnames'], np.loadtxt(fn, skiprows=data_args['skip'], unpack=True)))
                validation_data[obs]['label'] = data_args['label']
                validation_data[obs]['colnames'] = data_args['colnames']
            
            print(validation_data)
        return validation_data

        
    def plot_bias_results(self, corr_data, corr_theory, bias, z, catalog_name, output_dir,
                          err=None, chisq=None, mag_label=''):
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [5, 3]})
        colors = plt.cm.plasma_r(np.linspace(0.1, 1, len(self.test_samples))) # pylint: disable=no-member

        for sample_name, color in zip(self.test_samples, colors):
            sample_corr = corr_data[sample_name]
            sample_label = self.test_sample_labels.get(sample_name)
            sample_th = corr_theory[sample_name]
            ax[0].loglog(sample_corr[0], sample_th, c=color)
            _, caps, bars = ax[0].errorbar(sample_corr[0], sample_corr[1], sample_corr[2], marker='o', ls='', c=color,
                                                 label=sample_label)
            # add transparency for error bars
            [bar.set_alpha(0.2) for bar in bars]
            [cap.set_alpha(0.2) for cap in caps]
            #add shaded band for fit range
            ax[0].fill_between([self.fit_range[sample_name]['min_theta'], self.fit_range[sample_name]['max_theta']],
                                [self.fig_ylim[0], self.fig_ylim[0]], [self.fig_ylim[1], self.fig_ylim[1]],
                                alpha=0.07, color='grey')
            
        if self.title_in_legend:
            lgnd_title = '{}\n$({})$'.format(catalog_name, mag_label)
            title = self.data_label
        else:
            lgnd_title = '({})'.format(mag_label)
            title = '{} vs. {}'.format(catalog_name, self.data_label)
        ax[0].legend(loc='lower left', title=lgnd_title, framealpha=0.5,
                     fontsize=self.legend_size, title_fontsize=self.title_fontsize)
        ax[0].set_xlabel(self.fig_xlabel, size=self.font_size)
        ax[0].set_ylim(*self.fig_ylim)
        ax[0].set_ylabel(self.fig_ylabel, size=self.font_size)
        ax[0].set_title(title, fontsize='medium')

        ax[1].errorbar(z, bias, err, marker='o', ls='', label='{}\n$({})$'.format(catalog_name, mag_label))
        if not self.observations:
            ax[1].plot(z, bias) #plot curve through points
        #add validation data
        for v in self.validation_data.values():
            colz = v['colnames'][0]
            colb = 'bias'
            zmask = (v[colz] < np.max(z)*1.25)
            ax[1].plot(v[colz][zmask], v[colb][zmask], label=v['label'])
            print(v['colnames'])
            if 'b_lo'in v['colnames'] and 'b_hi' in v['colnames']:
                print('band', v[colz][zmask], v['b_lo'][zmask], v['b_hi'][zmask])
                ax[1].fill_between(v[colz][zmask], v['b_lo'][zmask], v['b_hi'][zmask], alpha=.3, color='grey')
        ax[1].set_title('Bias vs redshift', fontsize='medium')
        ax[1].set_xlabel('$z$', size=self.font_size)
        ax[1].set_ylabel('$b(z)$', size=self.font_size)
        ax[1].legend(loc='upper right', framealpha=0.5, frameon=True, fontsize=self.legend_size-2)
        if chisq:
            ax[1].text(0.95, 0.05, '$\chi^2/\\rm{{d.o.f}}={}$'.format(', '.join(['{:.2g}'.format(c) for c in chisq])),
                       horizontalalignment='right', verticalalignment='bottom',
                       transform=ax[1].transAxes)
        plt.subplots_adjust(wspace=.05)
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

        if self.truncate_cat_name:
            catalog_name = catalog_name.partition("_")[0]

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
        best_fit_err = []
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
            #extract covariance matrix
            #use curve_fit to get error on fit which has documented normalization for covariance matrix
            cfit = op.curve_fit(wtheta, w_th[angles], xi[angles], p0=1.0, sigma=xi_sig[angles], bounds=(0.1, 10))
            #best_bias_obj = result.hess_inv*np.identity(1)[0] #unknown relative normalization
            best_bias_err = np.sqrt(cfit[1][0][0])
            correlation_theory[sample_name] = best_bias**2*w_th
            best_fit_bias.append(best_bias[0])
            best_fit_err.append(best_bias_err)
            print(sample_name, best_fit_bias, w_th[angles], xi[angles], xi_sig[angles])

        z_mean = np.array(z_mean)
        best_fit_bias = np.array(best_fit_bias)
        best_fit_err = np.array(best_fit_err)
        chi_2 = []
        # compute chi*2 between best_fit bias and validation data
        for v in self.validation_data.values():
            colz = v['colnames'][0]
            colb = 'bias'
            validation_data = np.interp(z_mean, v[colz], v[colb])
            if 'b_lo'in v['colnames'] and 'b_hi' in v['colnames']:
                val_err_lo = np.abs(np.interp(z_mean, v[colz], v['b_lo']) - validation_data)
                val_err_hi = np.abs(np.interp(z_mean, v[colz], v['b_hi']) - validation_data)
                print(val_err_lo, val_err_hi)
                val_err = (val_err_lo + val_err_hi)/2 # mean of upper and lower errors
                error_sq = best_fit_err**2 + val_err**2 # sum in quadrature
                print(val_err, best_fit_err, error_sq)
            else:
                error_sq = best_fit_err**2
            chi__2 = np.sum((best_fit_bias - validation_data)**2/error_sq/len(best_fit_bias))
            print('\nchi**2(linear bias - bias data)={:.3g}'.format(chi__2))
            chi_2.append(chi__2)

        # get mag_cut for plot
        mag_label = ''
        if 'mag' in self.requested_columns.keys():
            filt = self.requested_columns['mag'][0].split('_')[1]
            mag_vals = self.test_samples[list(self.test_samples.keys())[0]]['mag'] # assume all cuts the same 
            #mag_label = '{:.2g} < {}'.format(mag_vals['min'], filt) if 'min' in mag_vals.keys() else filt
            mag_label = filt + ' < {:.3g}'.format(mag_vals['max'])
        self.plot_bias_results(corr_data=correlation_data,
                               catalog_name=catalog_name,
                               corr_theory=correlation_theory,
                               bias=best_fit_bias,
                               z=z_mean, mag_label=mag_label,
                               output_dir=output_dir,
                               err=best_fit_err, chisq=chi_2)

        passed = np.all((best_fit_bias[1:]-best_fit_bias[:-1]) > 0) 
        score = np.count_nonzero((best_fit_bias[:-1]-best_fit_bias[1:])>0)*1.0/(len(best_fit_bias)-1.0)
        return TestResult(score=score, passed=passed,
                  summary="Resulting linear bias obtained from the 2pcf")
