
from __future__ import print_function, division, unicode_literals, absolute_import
import os
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from itertools import count
import numpy as np
from GCR import GCRQuery
from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import get_sky_volume, get_opt_binpoints

__all__ = ['StellarMassFunction']


class StellarMassFunction(BaseValidationTest):
    """
    validation test to show N(z) distributions
    """
    #setup dict with parameters needed to read in validation data
    possible_observations = {
        'PRIMUS_2013': {
            'filename_template': 'SMF/moustakas_et_al_2013/Table{}.txt',
            'file-info': {
                '0. < z < .1':  {'zlo':0.,  'zhi':.1,  'table#':3, 'usecols':[0,1,2,3]},
                '.2 < z < .3':  {'zlo':.2,  'zhi':.3,  'table#':4, 'usecols':[0,1,2,3]},
                '.3 < z < .4':  {'zlo':.3,  'zhi':.4,  'table#':4, 'usecols':[0,6,7,8]},
                '.4 < z < .5':  {'zlo':.4,  'zhi':.5,  'table#':4, 'usecols':[0,11,12,13]},
                '.5 < z < .65': {'zlo':.5,  'zhi':.65, 'table#':4, 'usecols':[0,16,17,18]},
                '.65 < z < .8': {'zlo':.65, 'zhi':.8,  'table#':4, 'usecols':[0,21,22,23]},
                '.8 < z < 1.0': {'zlo':.8,  'zhi':1.0, 'table#':4, 'usecols':[0,26,27,28]},
            },
            'zrange': (0.0, 1.0),
            'colnames': ('logM', 'log_phi', 'dlog_phi+', 'dlog_phi-'),
            'label': 'PRIMUS 2013',
            'missingdata': '...',
        },
    }
    zkey_match = '< z <'

    #plotting constants
    validation_color = 'black'
    validation_marker = 'o'
    default_markers = ['v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']
    msize = 4 #marker-size
    yaxis_xoffset = 0.02
    yaxis_yoffset = 0.5

    def __init__(self, z='redshift_true', mass='stellar_mass', Nbins=25, log_Mlo=8., log_Mhi=12.,
                 observation='', zlo=0., zhi=1.0, zint=0.2, ncolumns=2, **kwargs):
        #pylint: disable=W0231

        self.font_size = kwargs.get('font_size', 20)
        self.legend_size = kwargs.get('legend_size', 13)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.fig_xsize = kwargs.get('fig_xsize', 10)
        self.fig_ysize = kwargs.get('fig_ysize', 14)
        self.text_size = kwargs.get('text_size', 20.)
        
        #catalog quantities
        self.zlabel = z
        self.Mlabel = mass

        #z-range  and mass binning
        self.Nbins = Nbins
        self.log_Mlo = log_Mlo
        self.log_Mhi = log_Mhi
        self.Mbins = np.logspace(log_Mlo, log_Mhi, Nbins+1)
        self.DM = (log_Mhi - log_Mlo)/Nbins

        #validation data
        self.validation_data = {}
        self.observation = observation

        #check for valid observations
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        elif observation not in self.possible_observations:
            raise ValueError('Observation {} not available'.format(observation))
        else:
            self.validation_data = self.get_validation_data(observation)

        #plotting variables
        self.ncolumns = int(ncolumns)
        self._color_iterator = ('C{}'.format(i) for i in count())

        #setup subplot configuration and get z cuts for each plot
        self.z_lo, self.z_hi = self.init_plots(zlo, zhi, zint)
        zmin = np.min(self.z_lo)
        zmax = np.max(self.z_hi)
        self.filters = [(lambda z: (z > zmin) & (z < zmax), self.zlabel)]

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns, sharex='col',
                                                         figsize=(self.fig_xsize, self.fig_ysize))
        self.summary_fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical',
                              fontsize=self.text_size) #setup a common axis label
        #could plot summary validation data here if available but would need to evaluate labels, bin values etc.
        #otherwise setup a check so that validation data is plotted only once on summary plot
        self.first_pass = True

        self._other_kwargs = kwargs


    def init_plots(self, zlo, zhi, zint):
        #get magnitude cuts based on validation data or default limits (only catalog data plotted)
        if not self.validation_data:
            z_lo = np.arange(zlo, zhi, zint)
            z_hi = np.arange(zint, zhi+zint, zint)
        else:
            z_lo = [self.validation_data[k].get('zlo') for k in self.validation_data.keys() if self.zkey_match in k]
            z_hi = [self.validation_data[k].get('zhi') for k in self.validation_data.keys() if self.zkey_match in k]

        print(z_lo, z_hi)
        #setup number of plots and number of rows required for subplots
        self.nplots = len(z_lo)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns

        #other plotting variables
        self.markers = iter(self.default_markers)
        self.yaxis = r'$d\phi/d\log_{10}(M/M_\odot)\quad[Mpc^{-3} dex^{-1}]$'
        self.xaxis = '$M^*/M_\\odot$'

        return z_lo, z_hi


    def get_validation_data(self, observation):
        data_args = self.possible_observations[observation]
        validation_data = {'label':data_args['label']}

        file_args = data_args['file-info']
        for zkey in file_args.keys():
            filename = self.possible_observations[observation]['filename_template'].format(file_args[zkey]['table#'])
            data_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(data_path):
                raise ValueError("SMF data file {} not found".format(data_path))
            if not os.path.getsize(data_path):
                raise ValueError("SMF data file {} is empty".format(data_path))

            data = np.genfromtxt(data_path, unpack=True, usecols=file_args[zkey]['usecols'], missing_values=data_args['missingdata'])
            validation_data[zkey] = dict(zip(data_args['colnames'], data))
            validation_data[zkey]['zlo'] = file_args[zkey]['zlo']
            validation_data[zkey]['zhi'] = file_args[zkey]['zhi']

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        #check catalog data for required quantities
        if not catalog_instance.has_quantities([self.zlabel, self.Mlabel]):
            return TestResult(skipped=True, summary='Missing required quantity {} or {}'.format(self.zlabel, self.Mlabel))

        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns, sharex='col', figsize=(self.fig_xsize, self.fig_ysize))
        fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical',
                 fontsize=self.text_size) #setup a common axis label
        if self.truncate_cat_name:
            catalog_name = catalog_name.partition("_")[0]
        
        #initialize arrays for storing histogram sums
        N_array = np.zeros((self.nrows, self.ncolumns, len(self.Mbins)-1), dtype=np.int)
        sumM_array = np.zeros((self.nrows, self.ncolumns, len(self.Mbins)-1))
        sumM2_array = np.zeros((self.nrows, self.ncolumns, len(self.Mbins)-1))

        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        for catalog_data in catalog_instance.get_quantities([self.zlabel, self.Mlabel], filters=self.filters, return_iterator=True):
            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)
            for cut_lo, cut_hi, N, sumM, sumM2 in zip_longest(
                self.z_lo,
                self.z_hi,
                N_array.reshape(-1, N_array.shape[-1]), #flatten all but last dimension of array
                sumM_array.reshape(-1, sumM_array.shape[-1]),
                sumM2_array.reshape(-1, sumM2_array.shape[-1]),
            ):
                if cut_lo is not None:  #cut_lo can be 0. so cannot use if cut_lo
                    mask = (cut_lo < catalog_data[self.zlabel]) & (catalog_data[self.zlabel] < cut_hi)
                    M_this = catalog_data[self.Mlabel][mask]
                    del mask

                    #bin catalog_data and accumulate subplot histograms
                    N += np.histogram(M_this, bins=self.Mbins)[0]
                    sumM += np.histogram(M_this, bins=self.Mbins, weights=M_this)[0]
                    sumM2 += np.histogram(M_this, bins=self.Mbins, weights=M_this**2)[0]

        #check that catalog has entries for quantity to be plotted
        if not np.asarray([N.sum() for N in N_array]).sum():
            raise ValueError('No data found for quantity {}'.format(self.Mlabel))

        #loop over magnitude cuts and make plots
        results = {}
        for n, (ax_this, summary_ax_this, cut_lo, cut_hi, N, sumM, sumM2, zkey) in enumerate(zip_longest(
            ax.flat,
            self.summary_ax.flat,
            self.z_lo,
            self.z_hi,
            N_array.reshape(-1, N_array.shape[-1]),
            sumM_array.reshape(-1, sumM_array.shape[-1]),
            sumM2_array.reshape(-1, sumM2_array.shape[-1]),
            [k for k in self.validation_data.keys() if self.zkey_match in k],
        )):
            if cut_lo is None:  #cut_lo is None if self.z_lo is exhausted
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)
            else:
                if not zkey:
                    zkey = '{:.1f} < z < {:.1f}'.format(cut_lo, cut_hi)
                cut_label = '${}$'.format(zkey)
                Mvalues = sumM/N
                sumN = N.sum()
                total = '(# of galaxies = {})'.format(sumN)
                Nerrors = np.sqrt(N)
                volume = get_sky_volume(catalog_instance.sky_area, cut_lo, cut_hi, catalog_instance.cosmology)
                phi = N/volume/self.DM
                phi_errors = Nerrors/volume/self.DM

                #make subplot
                validation_label = self.validation_data.get('label', '')
                results[zkey] = {'Mphi':Mvalues, 'total':total, 'phi':phi, 'phi+-':phi_errors}
                self.catalog_subplot(ax_this, Mvalues, phi, phi_errors, catalog_color, catalog_name)
                if zkey in self.validation_data.keys():
                    data = self.validation_subplot(ax_this, self.validation_data[zkey], validation_label)
                    results[zkey].update(data)
                self.decorate_subplot(ax_this, n, label=cut_label)

                #add curve for this catalog to summary plot
                self.catalog_subplot(summary_ax_this, Mvalues, phi, phi_errors, catalog_color, catalog_name)
                if self.first_pass and zkey in self.validation_data.keys():  #add validation data if evaluating first catalog
                    self.validation_subplot(summary_ax_this, self.validation_data[zkey], validation_label)
                self.decorate_subplot(summary_ax_this, n, label=cut_label)


        #save results for catalog and validation data in txt files
        for filename, dtype, info in zip_longest((catalog_name, self.observation), ('phi', 'data'), ('total',)):
            if filename:
                with open(os.path.join(output_dir, 'SMF_' + filename + '.txt'), 'ab') as f_handle: #open file in append mode
                    #loop over magnitude cuts in results dict
                    for key, value in results.items():
                        self.save_quantities(dtype, value, f_handle, comment=' '.join((key, value.get(info, ''))))

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, 'SMF_' + catalog_name + '.png'))
        plt.close(fig)
        return TestResult(inspect_only=True)


    def catalog_subplot(self, ax, M, phi, phi_errors, catalog_color, catalog_label):
        ax.plot(M, phi, label=catalog_label, color=catalog_color)
        ax.fill_between(M, phi - phi_errors, phi + phi_errors, alpha=0.3, facecolor=catalog_color)


    def validation_subplot(self, ax, validation_data, validation_label):
        results = dict()
        if all(x in validation_data.keys() for x in ('logM', 'log_phi', 'dlog_phi+', 'dlog_phi-')):
            M = np.power(10, validation_data['logM'])
            phi = np.power(10, validation_data['log_phi'])
            dphi_hi = np.power(10, validation_data['log_phi'] + validation_data['dlog_phi+']) - phi
            dphi_lo = -np.power(10, validation_data['log_phi'] + validation_data['dlog_phi-']) + phi
            ax.errorbar(M, phi, yerr=[dphi_lo, dphi_hi], color=self.validation_color, marker=self.validation_marker,
                        linestyle="", label=validation_label, ms=self.msize)
            results['data'] = phi
            results['Mdata'] = M
            results['data+'] = dphi_hi
            results['data-'] = dphi_lo
        else:
            raise ValueError("Missing expected validation-data quantitites")

        return results


    def decorate_subplot(self, ax, nplot, label=None):
        ax.tick_params(labelsize=14)
        ax.set_xscale('log')
        ax.set_yscale('log')
        if label:
            ax.text(0.95, 0.95, label, horizontalalignment='right', verticalalignment='top',
                    fontsize=self.text_size, transform=ax.transAxes)

        #add axes and legend
        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            #print "noticks",nplot
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(False)
                #prevent overlapping yaxis labels
                ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            ax.set_xlabel(self.xaxis, size=self.font_size)
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(True)
            ax.xaxis.set_tick_params(which='major', labelbottom=True)
        ax.legend(loc='lower left', fancybox=True, framealpha=0.5, numpoints=1, fontsize=self.legend_size)


    @staticmethod
    def post_process_plot(fig):
        fig.subplots_adjust(hspace=0)


    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('M'+keyname, keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <M>', keyname, keyname+'-', keyname+'+', ' '))
            elif keyname+'+-' in results:
                fields = ('M'+keyname, keyname, keyname+'+-')
                header = ', '.join(('Data columns are: <M>', keyname, keyname+'+-', ' '))
            else:
                fields = ('M'+keyname, keyname)
                header = ', '.join(('Data columns are: <M>', keyname, ' '))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
