from __future__ import print_function, division, unicode_literals, absolute_import
import os
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from itertools import count

import re
import numpy as np
from scipy.interpolate import interp1d
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['EllipticityDistribution']

class EllipticityDistribution(BaseValidationTest):
    """
    validation test to show total ellipticity distributions
    """
    #setup dict with parameters needed to read in validation data
    possible_observations = {
        'COSMOS_2013': {
            'label': 'COSMOS 2013',
            'band_mag': 'i',
            'band_Mag': ['V', 'r', 'g'],
            'zlo': 0.0,
            'zhi': 2.0,
            'definition': 'e_distortion',
            'morphology': ('LRG', 'early', 'disk', 'late'),
            'filename_template': 'ellipticity/COSMOS/joachimi_et_al_2013/{}{}_{}.dat',
            'file-info': {
                'LRG':{'prefix':'all', 'suffix':'mag24'},
                'early':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'disk':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'late':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'irregular':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'usecols':0,
                'skiprows':0,
            },
            'cuts':{
                'LRG':{'B/T_min':0.7, 'B/T_max':1., 'mag_lo':24., 'Mag_hi':-np.inf, 'Mag_lo':-19.},
                'early':{'B/T_min':0.7, 'B/T_max':1.0, 'mag_lo':24., 'Mag_hi':-21., 'Mag_lo':-17.},
                'disk':{'B/T_min':0., 'B/T_max':0.2, 'mag_lo':24., 'Mag_hi':-21., 'Mag_lo':-17.},
                'late':{'B/T_min':0.4, 'B/T_max':0.7, 'mag_lo':24., 'Mag_hi':-21., 'Mag_lo':-17.},
                'irregular':{'B/T_min':0.0, 'B/T_max':1.0},
                'ancillary_quantities':['bulge_to_total_ratio_i', 'bulge_to_total_ratio_stellar',
                                        'bulge_to_total_ratio'],
                'ancillary_keys':['B/T'],
            },
        },
    }
    
    #define ellipticity functions
    @staticmethod
    def e_shear(e):
        return e

    @staticmethod
    def e_distortion(a, b):
        q = b/a
        return  (1-q**2)/(1+q**2)

    @staticmethod
    def e_shear_to_distortion(e):
        return 2*e/(1+e**2)

    #plotting constants
    lw2 = 2
    fsize = 16
    lsize = 10
    validation_color = 'black'
    validation_marker = 'o'
    default_markers = ['v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']
    msize = 4 #marker-size
    yaxis_xoffset = 0.02
    yaxis_yoffset = 0.5

    def __init__(self, z='redshift_true', zlo=0., zhi=2., N_ebins=40, observation='', ncolumns=2,
                 morphology=('all',), band_mag='i', mag_lo=24, band_Mag='r', Mag_hi=-21, Mag_lo=-17, normed=False,
                 **kwargs):
        #pylint: disable=W0231
        #catalog quantities
        self.filter_quantities = [z]
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.title_in_legend = kwargs.get('title_in_legend', False)
        self.legend_location = kwargs.get('legend_location', 'lower left')
        self.xfont_size = kwargs.get('xfont_size', 12)
        self.yfont_size = kwargs.get('yfont_size', 14)
        self.legend_size = kwargs.get('legend_size', 6)
        self.legend_title_size = kwargs.get('legend_title_size', 8)
        self.catalog_ellipticity_definition = kwargs.get('ellipticity_definition', 'e_shear')
        self.convert_to_distortion = kwargs.get('convert_to_distortion', False)

        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                               'mag_true_{}_lsst',
                               'mag_true_{}_sdss',
                               'mag_true_{}_des',
                              )
        possible_Mag_fields = ('Mag_true_{}_z0',
                               'Mag_true_{}_lsst_z0',
                               'Mag_true_{}_sdss_z0',
                               'Mag_true_{}_des_z0',
                              )

        possible_native_luminosities = {'V':'otherLuminosities/totalLuminositiesStellar:V:rest',
                                       }

        possible_ellipticity_definitions = {'e_shear':{'possible_quantities':[['ellipticity', 'ellipticity_true']],
                                                       'function':self.e_shear,
                                                       'xaxis_label': r'$e = (1-q)/(1+q)$',
                                                       'file_label':'es',
                                                      },
                                            'e_distortion':{'possible_quantities':[['size', 'size_true'], ['size_minor', 'size_minor_true']],
                                                            'function':self.e_distortion,
                                                            'xaxis_label': r'$e = (1-q^2)/(1+q^2)$',
                                                            'file_label':'ed',
                                                           },
                                           }
        #binning
        self.N_ebins = N_ebins
        self.ebins = np.linspace(0., 1, N_ebins+1)

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
        self.normed = normed

        #morphologies
        self.morphology = self.validation_data.get('morphology', morphology)

        #cuts
        self.zlo = self.validation_data.get('zlo', float(zlo))
        self.zhi = self.validation_data.get('zhi', float(zhi))
        self.filters = [(lambda z: (z > self.zlo) & (z < self.zhi), z)]
        self.band_mag = self.validation_data.get('band_mag', band_mag)
        self.possible_mag_fields = [f.format(self.band_mag) for f in possible_mag_fields]
        self.band_Mag = self.validation_data.get('band_Mag', [band_Mag])
        self.possible_Mag_fields = [f.format(band) for f in possible_Mag_fields for band in self.band_Mag]
        self.mag_lo = dict(zip(self.morphology, [self.validation_data.get('cuts', {}).get(m, {}).get('mag_lo', mag_lo) for m in self.morphology]))
        self.Mag_lo = dict(zip(self.morphology, [self.validation_data.get('cuts', {}).get(m, {}).get('Mag_lo', Mag_lo) for m in self.morphology]))
        self.Mag_hi = dict(zip(self.morphology, [self.validation_data.get('cuts', {}).get(m, {}).get('Mag_hi', Mag_hi) for m in self.morphology]))

        #check for ellipticity definitions
        self.possible_quantities = possible_ellipticity_definitions[self.validation_data.get('definition', 'e_shear')]['possible_quantities']
        self.ellipticity_function = possible_ellipticity_definitions[self.validation_data.get('definition', 'e_shear')].get('function')
        self.xaxis_label = possible_ellipticity_definitions[self.validation_data.get('definition', 'e_shear')].get('xaxis_label')
        self.file_label = possible_ellipticity_definitions[self.validation_data.get('definition', 'e_shear')].get('file_label')

        #check for ellipticity conversions and overwrite previous definitions
        if self.convert_to_distortion and self.catalog_ellipticity_definition == 'e_shear':
            self.possible_quantities = possible_ellipticity_definitions[self.catalog_ellipticity_definition]['possible_quantities']
            self.ellipticity_function = self.e_shear_to_distortion
            print('Converting from e_shear to e_distortion')

        #check for native quantities
        self.native_luminosities = dict(zip([band for band in possible_native_luminosities if band in self.band_Mag],\
                                            [possible_native_luminosities[band] for band in possible_native_luminosities if band in self.band_Mag]))

        #check for ancillary quantities
        self.possible_ancillary_quantities = self.validation_data.get('cuts', {}).get('ancillary_quantities', None)

        #setup subplot configuration
        self.init_plots()

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns, sharex='col')
        #could plot summary validation data here if available but would need to evaluate labels, bin values etc.
        #otherwise setup a check so that validation data is plotted only once on summary plot
        self.first_pass = True

        self.validation_percentiles = {
            'percentiles': kwargs['validation_percentile_points'],
            'ranges': kwargs['validation_percentile_ranges']}

        self._other_kwargs = kwargs


    def init_plots(self):
        #setup plots and determine number of rows required for subplots
        self.nplots = len(self.morphology)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns
        self._color_iterator = ('C{}'.format(i) for i in count())

        #other plotting variables
        self.markers = iter(self.default_markers)
        if self.normed:
            self.yaxis = '$P(e)$'
        else:
            self.yaxis = '$N$'


    def get_validation_data(self, observation):
        data_args = self.possible_observations[observation]

        validation_data = {}
        for m in data_args['morphology']:
            file_info = data_args['file-info'][m]
            data_path = os.path.join(self.data_dir, data_args['filename_template'].format(file_info['prefix'], m.lower(), file_info['suffix']))

            if not os.path.exists(data_path):
                raise ValueError("{} data file {} not found".format(m, data_path))

            if not os.path.getsize(data_path):
                raise ValueError("{} data file {} is empty".format(m, data_path))

            validation_data[m] = np.loadtxt(data_path, unpack=True, usecols=data_args['file-info']['usecols'],\
                                            skiprows=data_args['file-info']['skiprows'])

        #collect remaining information
        for arg in data_args:
            if not 'file' in arg:
                validation_data[arg] = data_args[arg]

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        #add quantities to catalog if needed
        for band in self.native_luminosities:
            if catalog_instance.has_quantity(self.native_luminosities[band]):
                catalog_instance.add_quantity_modifier('Mag_true_{}_z0'.format(band), (lambda x: -2.5*np.log10(x), self.native_luminosities[band]))

        print('Checking for required quantities')
        #check catalog data for required quantities
        required_quantities = []
        for pgroup in self.possible_quantities:
            found_quantity = catalog_instance.first_available(*pgroup)
            if found_quantity is not None:
                required_quantities.append(found_quantity)
        if not catalog_instance.has_quantities(required_quantities + self.filter_quantities):
            return TestResult(skipped=True, summary='Missing some required quantities: {}'.format(', '.join(required_quantities)))
        print('Required quantities', required_quantities)
        ancillary_quantity = None
        if self.possible_ancillary_quantities is not None:
            ancillary_quantity = catalog_instance.first_available(*self.possible_ancillary_quantities)
            if ancillary_quantity is None:
                return TestResult(skipped=True, summary='Missing some ancillary quantities: {}'.format(', '.join(self.possible_ancillary_quantities)))

        mag_field = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped=True, summary='Missing needed quantities to make magnitude cuts')
        Mag_field = catalog_instance.first_available(*self.possible_Mag_fields)
        if not Mag_field:
            return TestResult(skipped=True, summary='Missing needed quantities to make magnitude cuts')
        all_quantities = required_quantities +[mag_field, Mag_field] + self.filter_quantities
        if ancillary_quantity is not None:
            all_quantities = all_quantities + [ancillary_quantity]
        print('Fetching quantities', all_quantities)

        mag_filtername = str(mag_field.split('_')[-2])
        Mag_filtername = str(Mag_field.split('_')[2])
        filelabel = '_'.join(('m', mag_filtername, 'M', Mag_filtername))

        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns, sharex='col')
        fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical',
                 fontsize=self.yfont_size) #setup a common axis label

        #initialize arrays for storing histogram sums
        N_array = np.zeros((self.nrows, self.ncolumns, len(self.ebins)-1), dtype=int)
        sume_array = np.zeros((self.nrows, self.ncolumns, len(self.ebins)-1))
        sume2_array = np.zeros((self.nrows, self.ncolumns, len(self.ebins)-1))
        
        #initialize boolean values for checking ellipticity endpoints
        any_low = False
        any_high = False

        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]
        
        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        for catalog_data in catalog_instance.get_quantities(all_quantities, filters=self.filters, return_iterator=True):
            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)
            for morphology, N, sume, sume2 in zip_longest(
                    self.morphology,
                    N_array.reshape(-1, N_array.shape[-1]), #flatten all but last dimension of array
                    sume_array.reshape(-1, sume_array.shape[-1]),
                    sume2_array.reshape(-1, sume2_array.shape[-1]),
            ):
                #make cuts
                if morphology is not None:
                    mask = (catalog_data[mag_field] < self.mag_lo.get(morphology))
                    mask &= (self.Mag_hi.get(morphology) < catalog_data[Mag_field]) & (catalog_data[Mag_field] < self.Mag_lo.get(morphology))
                    if ancillary_quantity is not None:
                        for aq, key  in zip_longest([ancillary_quantity], self.validation_data['cuts'].get('ancillary_keys')):
                            mask &= (self.validation_data['cuts'][morphology].get(key+'_min') < catalog_data[aq]) &\
                                    (catalog_data[aq] < self.validation_data['cuts'][morphology].get(key+'_max'))

                    print('Number of {} galaxies passing selection cuts for morphology {} = {}'.format(catalog_name, morphology, np.sum(mask)))
                    #compute ellipticity from definition
                    e_this = self.ellipticity_function(*[catalog_data[q][mask] for q in required_quantities])
                    #print('mm', np.min(e_this), np.max(e_this))
                    del mask

                    #accumulate histograms
                    N += np.histogram(e_this, bins=self.ebins)[0]
                    sume += np.histogram(e_this, bins=self.ebins, weights=e_this)[0]
                    sume2 += np.histogram(e_this, bins=self.ebins, weights=e_this**2)[0]
                    
                    #check borders
                    if len(e_this)>0:
                        if np.min(e_this)<0:
                            any_low = True
                            print('Value<0 found for morphology {} in catalog {}: {}'.format(morphology, catalog_name, np.min(e_this)))
                        if np.max(e_this)>1:
                            any_high = True
                            print('Value>1 found for morphology {} in catalog {}: {}'.format(morphology, catalog_name, np.max(e_this)))

        #check that catalog has entries for quantity to be plotted
        if not np.asarray([N.sum() for N in N_array]).sum():
            raise ValueError('No data found for quantities {}'.format(', '.join(required_quantities)))

        #make plots
        results = {}
        n_fails = 0 + any_low + any_high
        for n, (ax_this, summary_ax_this, morphology, N, sume, sume2) in enumerate(zip_longest(
                ax.flat,
                self.summary_ax.flat,
                self.morphology,
                N_array.reshape(-1, N_array.shape[-1]), #flatten all but last dimension of array
                sume_array.reshape(-1, sume_array.shape[-1]),
                sume2_array.reshape(-1, sume2_array.shape[-1]),
        )):
            if morphology is not None:
                #get labels
                cutlabel = '${} < {} < {}$; ${} < {}$'.format(str(self.Mag_hi.get(morphology)), Mag_filtername, str(self.Mag_lo.get(morphology)),\
                                                              mag_filtername, str(self.mag_lo.get(morphology)))
                cutlabel = re.sub('-inf < ', '' , cutlabel) #truncate label with inf

                ancillary_label = []
                if ancillary_quantity is not None:
                    for key  in self.validation_data['cuts'].get('ancillary_keys'):
                        ancillary_label.append('${}<{}<{}$'.format(str(self.validation_data['cuts'][morphology].get(key+'_min')),\
                                               key, str(self.validation_data['cuts'][morphology].get(key+'_max'))))
                ancillary_label = '; '.join(ancillary_label)
                catalog_label = '; '.join((catalog_name, ancillary_label))
                validation_label = ' '.join((self.validation_data.get('label', ''), morphology))
                reskey = cutlabel.replace('$', '')

                #get points to be plotted
                e_values = sume/N
                sumN = N.sum()
                total = '(# of galaxies = {})'.format(sumN)
                Nerrors = np.sqrt(N)
                if self.normed:
                    binwidths = self.ebins[1:] - self.ebins[:-1]
                    N = N/sumN/binwidths
                    Nerrors = Nerrors/sumN/binwidths

                results[reskey] = {'catalog':{'e_ave':e_values, 'N':N, 'N+':N+Nerrors, 'N-':N-Nerrors,\
                                   'total':total, 'xtralabel':ancillary_label.replace('$', '')}}
                self.catalog_subplot(ax_this, e_values, N, catalog_color, catalog_label)
                results[reskey]['validation'] = self.validation_subplot(ax_this, self.validation_data.get(morphology), validation_label)
                self.decorate_subplot(ax_this, n, label=cutlabel)

                #add curve for this catalog to summary plot
                self.catalog_subplot(summary_ax_this, e_values, N, catalog_color, catalog_label, errors=Nerrors)
                if self.first_pass: #add validation data if evaluating first catalog
                    self.validation_subplot(summary_ax_this, self.validation_data.get(morphology), validation_label)
                self.decorate_subplot(summary_ax_this, n, label=cutlabel)
                
                #check ellipticity distributions
                number_passed, percentiles = self.validate_percentiles(N)
                print("Percentiles for morphology {} are: ".format(morphology)+', '.join([" {:.3f} ({})".format(p, v) for p,v in zip(percentiles, self.validation_percentiles['percentiles'])]))
                if number_passed>0:
                    print("Ellipticity percentile check failed for morphology {}".format(morphology))
                n_fails += number_passed
                

            else:
                #make empty subplots invisible
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)
        
        #check overall ellipticity distribution
        global_N = np.sum(np.sum(N_array, axis=1), axis=0)
        number_passed, percentiles = self.validate_percentiles(global_N)
        print("Percentiles for global distribution are: "+', '.join([" {:.3f} ({})".format(p, v) for p,v in zip(percentiles, self.validation_percentiles['percentiles'])]))
        if number_passed>0:
            print("Ellipticity percentile check failed for global distribution")
        n_fails += number_passed
        


        #save results for catalog and validation data in txt files
        for filename, dkey, dtype, info in zip_longest((catalog_name, self.observation), ('catalog', 'validation'), ('N', 'data'), ('total',)):
            if filename:
                with open(os.path.join(output_dir, ''.join(['Nvs', self.file_label, '_', filelabel+'.txt'])), 'ab') as f_handle: #open file in append mode
                    #loop over cuts in results dict
                    for key, value in results.items():
                        self.save_quantities(dtype, value[dkey], f_handle, comment=' '.join((key, value[dkey].get('xtralabel', ''), value[dkey].get(info, ''))))

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, ''.join(['Nvs', self.file_label, '_', filelabel+'.png'])))
        plt.close(fig)
        return TestResult(score=n_fails, passed=(n_fails==0))


    def catalog_subplot(self, ax, e_values, N, catalog_color, catalog_label, errors=None):
        ax.plot(e_values, N, label=catalog_label, color=catalog_color)
        if errors is not None:
            ax.fill_between(e_values, N+errors, N-errors, alpha=0.3, facecolor=catalog_color)


    def validation_subplot(self, ax, validation_data, validation_label):
        results = dict()
        if validation_data is not None:
            N, _ = np.histogram(validation_data, bins=self.ebins)
            sum_e, _ = np.histogram(validation_data, bins=self.ebins, weights=validation_data)
            e_ave = sum_e/N
            errors = np.sqrt(N)
            if self.normed:
                sumN = N.sum()
                binwidths = self.ebins[1:] - self.ebins[:-1]
                N = N/sumN/binwidths
                errors = errors/sumN/binwidths
            ax.errorbar(e_ave, N, yerr=errors, color=self.validation_color, label=validation_label, marker=self.validation_marker)
            results['e_ave'] = e_ave
            results['data'] = N
            results['data+'] = N + errors
            results['data-'] = N - errors

        return results


    def decorate_subplot(self, ax, nplot, label=None):
        ax.tick_params(labelsize=8)
        ax.set_yscale('log')
        if label and not self.title_in_legend:
            ax.set_title(label, fontsize='x-small')

       #add axes and legend
        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(False)
                #prevent overlapping yaxis labels
                ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            ax.set_xlabel(self.xaxis_label, size=self.xfont_size)
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(True)
        legend = ax.legend(loc=self.legend_location, fancybox=True, framealpha=0.5, numpoints=1, fontsize=self.legend_size)
        if self.title_in_legend: 
            legend.set_title(label, prop = {'size':self.legend_title_size})

    def validate_percentiles(self, data):
        cdf = np.zeros(self.N_ebins+1)
        for i in range(self.N_ebins):
            cdf[i+1:] += data[i]
        cdf /= cdf[-1]
        cdf *= 100 # because numpy percentile wants percentages
        interpolator = interp1d(cdf, self.ebins)
        percentiles = interpolator(self.validation_percentiles['percentiles'])
        return np.sum([(p<=pmin) | (p>=pmax) for p, (pmin, pmax) in zip(
                            percentiles, self.validation_percentiles['ranges'])]), percentiles
            

    def post_process_plot(self, fig):
        if self.title_in_legend:
            fig.subplots_adjust(hspace=0)

    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('e_ave', keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <e>', keyname, keyname+'-', keyname+'+', ' '))
            else:
                fields = ('e_ave', keyname)
                header = ', '.join(('Data columns are: <e>', keyname, ' '))
            np.savetxt(filename, np.vstack([results[k] for k in fields]).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
