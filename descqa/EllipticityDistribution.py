from __future__ import print_function, division, unicode_literals, absolute_import
import os
import math
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

import numpy as np
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
            'label': 'Joachimi et. al. 2013',
            'band_mag':'i',
            'band_Mag':['V', 'r', 'g'],
            'zrange': (0.0, 2.0),
            'definition':'e_qsquared'
            #'morphology':('lrg','early','disk', 'late', 'irregular'),
            'morphology':('lrg','early','disk', 'late'),
            'filename_template': 'ellipticity/COSMOS/joachimi_et_al_2013/{}{}_{}.dat',
            'file-info': {
                'lrg':{'prefix':'all', 'suffix':'mag24'},
                'early':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'disk':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'late':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
                'irregular':{'prefix':'histo_', 'suffix':'mag24_V17-21'},
            },
            'colnames':('ellipticity'),
            'usecols':0,
            'skiprows':0,
            'cuts':{
                'lrg':{'B/T_min':0.7, 'B/T_max':1., 'mag_lo':24., 'Mag_hi':-27., 'Mag_lo':-20.},
                'early':{'B/T_min':0.7, 'B/T_max':1.0, 'mag_lo':24., 'Mag_hi':-27., 'Mag_lo':-17.},
                'disk':{'B/T_min':0., 'B/T_max':0.2, 'mag_lo':24., 'Mag_hi':-27., 'Mag_lo':-17.},
                'late':{'B/T_min':0.4, 'B/T_max':0.7, 'mag_lo':24., 'Mag_hi':-27., 'Mag_lo':-17.},
                'irregular':{'B/T_min':None, 'B/T_max':None},
            },
        },
    }
    possible_ellipticity_defintions = {'default':{'quantities':['ellipticity', 'ellipticity_true']},
                                       'e_squared':{'quantities':['size', 'size_true', 'size_minor', 'size_minor_true'],
                                                    'function':e_squared,
                                                   },
                                      }
    ancillary_quantities = {'B/T': 'bulge_to_total_ratio',
                           } 
    def e_qsquared(q):
        return  np.sqrt((1-q**2)/(1+q**2))

    #plotting constants
    lw2 = 2
    fsize = 16
    lsize = 10
    validation_color = 'black'

    def __init__(self, e='ellipticity', z='redshift_true', zrange=(0.,2.),  N_ebins=40, N_theta_bins=20, observation='', ncolumns=2, 
                 morphology =['all'], band_mag='i', mag_lo=24, band_Mag='r', Mag_lo=-17, Mag_hi=-27,
                 **kwargs):

        #catalog quantities
        self.zlabel = z
        self.elabel = e
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                              )
        possible_Mag_fields = ('Mag_{}',
                               'Mag_{}_lsst',
                               'Mag_{}_sdss',
                               'Mag_{}_des',
                              )

        #binning
        self.N_ebins = N_ebins
        self.ebins = np.linspace(0., 1,, N_ebins+1)
        self.N_thetabins = N_theta_bins
        self.thetabins = np.linspace(-np.pi/2, np.pi/2, N_thetabins+1) #angle 

        #validation data
        self.validation_data = {}
        self.observation = observation

        #check for valid observations
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        elif observation not in possible_observations:
            raise ValueError('Observation {} not available'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)

        #plotting variables
        self.ncolumns = int(ncolumns)

        #cuts
        self.zrange = self.validation_data.get('zrange',zrange)
        self.filters = [(lambda z: (z > self.zrange[0]) & (z < self.zrange[1]), self.zlabel)]
        self.band_mag = self.validation_data.get('band_mag',band_mag)
        self.possible_mag_fields = [f.format(self.band_mag) for f in possible_mag_fields]
        self.band_Mag = self.validation_data.get('band_Mag',[band_Mag])
        self.possible_Mag_fields = [f.format(band) for f in possible_mag_fields for band in self.band_Mag]
        self.mag_lo = dict(zip(self.morphology, [self.validation_data['cuts'][m].get('mag_lo', mag_lo) for m in self.morphology]))
        self.Mag_lo = dict(zip(self.morphology, [self.validation_data['cuts'][m].get('Mag_lo', Mag_lo) for m in self.morphology]))
        self.Mag_hi = dict(zip(self.morphology, [self.validation_data['cuts'][m].get('Mag_hi', Mag_hi) for m in self.morphology]))
        print(self.mag_lo)

        #check for alternate ellipticity definitions
        if not self.validation_data.get('definition', None):
            self.required_quantities = self.possible_ellipticity_definitions[self.validation_data.get('definition')]['quantities'].append(self.zlabel)
        else:
            self.required_quantities = self.possible_ellipticity_definitions['default']['quantities'].append(self.zlabel)
        print('q',self.required_quantities)

        #setup subplot configuration
        self.init_plots()

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p))
        #could plot summary validation data here if available but would need to evaluate labels, bin values etc.
        #otherwise setup a check so that validation data is plotted only once on summary plot
        self.first_pass = True

        self._other_kwargs = kwargs


    def init_plots(self):
        #setup plots and determine number of rows required for subplots
        self.nplots = len(self.morphology)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns
        self._color_iterator = ('C{}'.format(i) for i in count())

        #other plotting variables
        self.markers = iter(self.default_markers)
        self.yaxis = '$P(e)$'
        self.xaxis = '$e$'
        return


    def get_validation_data(self, observation):
        data_args = possible_observations[observation]

        for m in data_args['morphology']:
            file_info = data_args['file_info'][m]
            data_path = os.path.join(self.data_dir, data_args['filename_template'].format(file_info[m]['prefix'], m, file_info[m]['suffix']))

            if not os.path.exists(data_path):
                raise ValueError("{} data file {} not found".format(m, data_path))

            if not os.path.getsize(data_path):
                raise ValueError("{} data file {} is empty".format(m, data_path))

            data = np.loadtxt(data_path, unpack=True, usecols=data_args['usecols'], skiprows=data_args['skiprows'])

            validation_data[m] = dict(zip(data_args['colnames'], data))

        #collect remaining information
        for arg in data_args:
            validation_data[arg] = data_args[arg] if not 'file' in arg
        print(validation_data.keys())

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        #check catalog data for required quantities
        if not catalog_instance.has_quantities([self.zlabel, self.elabel]):
            return TestResult(skipped=True, summary='Missing required quantity {} or {}'.format(self.zlabel, self.elabel))
        mag_field = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped=True, summary='Missing needed quantities to make magnitude cuts')
        Mag_field = catalog_instance.first_available(*self.possible_Mag_fields)
        if not Mag_field:
            return TestResult(skipped=True, summary='Missing needed quantities to make magnitude cuts')

        mag_filtername = mag_field.rpartition('_')[-1][-1].upper()
        Mag_filtername = Mag_field.rpartition('_')[-1][-1].upper()
        filelabel = '_'.join((mag_filtername, 'm'+self.band_mag, Mag_filtername, 'M'+self.band_Mag))

        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns, sharex='col')
        fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical') #setup a common axis label

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






        #loop over magnitude cuts and make plots
        results = {}
        for n, (ax_this, summary_ax_this, cut_lo, cut_hi, z0, z0err) in enumerate(zip_longest(
                ax.flat,
                self.summary_ax.flat,
                self.mag_lo,
                self.mag_hi,
                self.validation_data.get('z0values', []),
                self.validation_data.get('z0errors', []),
        )):
            if cut_lo is not None:
                mask = (catalog_data[mag_field] < cut_lo)
                cutlabel = '{} $< {}$'.format(self.band, cut_lo)
                if cut_hi:
                    mask &= (catalog_data[mag_field] >= cut_hi)
                    cutlabel = '${} <=$ '.format(cut_hi) + cutlabel #also appears in txt file so don't use \leq

                if z0 is None and 'z0const' in self.validation_data:
                    z0 = self.validation_data['z0const'] + self.validation_data['z0linear'] * cut_lo

                z_this = catalog_data[self.zlabel][mask]
                del mask
                total = '(# of galaxies = {})'.format(len(z_this))

                #bin catalog_data
                N = np.histogram(z_this, bins=self.zbins)[0]
                sumz = np.histogram(z_this, bins=self.zbins, weights=z_this)[0]
                meanz = sumz/N

                #make subplot
                catalog_label = ' '.join((catalog_name, cutlabel.replace(self.band, filtername + ' ' + self.band)))
                validation_label = ' '.join((self.validation_data.get('label', ''), cutlabel))
                reskey = cutlabel.replace('$', '')
                results[reskey] = self.make_subplot(ax_this, n, yaxis, z_this, meanz, z0, z0err, catalog_color, catalog_label, validation_label)
                results[reskey]['total'] = total

                #add curve for this catalog to summary plot
                if self.first_pass: #add validation data if evaluating first catalog
                    self.make_subplot(summary_ax_this, n, yaxis, z_this, meanz, z0, z0err, catalog_color, catalog_label, validation_label)
                else:
                    self.make_subplot(summary_ax_this, n, yaxis, z_this, meanz, None, None, catalog_color, catalog_label, validation_label)

            else:
                #make empty subplots invisible
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)

        #save results for catalog and validation data in txt files
        for filename, dtype, comment, info in zip_longest((filelabel, self.observation), ('y', 'fit'), (filtername,), ('total',)):
            if filename:
                with open(os.path.join(output_dir, 'Nvsz_' + filename + '.txt'), 'ab') as f_handle: #open file in append mode
                    #loop over magnitude cuts in results dict
                    for key, value in results.items():
                        self.save_quantities(dtype, value, f_handle, comment=' '.join(((comment or ''), key, value.get(info, ''))))

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, 'Nvsz_' + filelabel + '.png'))
        plt.close(fig)
        return TestResult(0, passed=True)


    def make_subplot(self, f, nplot, yaxis, catalog_data, meanz, z0, z0err, catalog_color, catalog_label, validation_label):
        #TODO: this function needs to be refactored

        if nplot % self.ncolumns == 0:  #1st column
            f.set_ylabel('$'+yaxis+'$', size=self.fsize)

        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            #print "noticks",nplot
            for axlabel in f.get_xticklabels():
                axlabel.set_visible(False)
                #prevent overlapping yaxis labels
                f.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            f.set_xlabel('$z$', size=fsize)
            for axlabel in f.get_xticklabels():
                axlabel.set_visible(True)

        #plot catalog data if available
        results = {'meanz': meanz}
        if len(catalog_data):
            results['y'] = f.hist(catalog_data, bins=self.zbins, label=catalog_label, color=catalog_color, lw=lw2, normed=self.normed, histtype='step')[0]

        #plot validation data if available
        if z0 and z0 > 0:
            if not self.normed:
                raise ValueError("Only fits to normed plots are implemented so far")

            ndata = meanz**2*np.exp(-meanz/z0)
            norm = self.nz_norm(self.zhi, z0) - self.nz_norm(self.zlo, z0)
            f.plot(meanz, ndata/norm, label=validation_label, ls='--', color=validation_color, lw=lw2)
            results['fit'] = ndata/norm

            if z0err and z0err > 0:
                nlo = meanz**2*np.exp(-meanz/(z0-z0err))
                nhi = meanz**2*np.exp(-meanz/(z0+z0err))
                normlo = self.nz_norm(self.zhi, z0-z0err) - self.nz_norm(self.zlo, z0-z0err)
                normhi = self.nz_norm(self.zhi, z0+z0err) - self.nz_norm(self.zlo, z0+z0err)
                f.fill_between(meanz, nlo/normlo, nhi/normhi, alpha=0.3, facecolor=validation_color)
                results['fit+'] = nhi/normhi
                results['fit-'] = nlo/normlo

        f.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=lsize, numpoints=1)

        return results


    @staticmethod
    def nz_norm(z, z0):
        return z0*math.exp(-z/z0)*(-z*z-2.*z*z0-2.*z0*z0)


    @staticmethod
    def post_process_plot(fig):
        fig.subplots_adjust(hspace=0)


    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('meanz', keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <z>', keyname, keyname+'-', keyname+'+', ' '))
            else:
                fields = ('meanz', keyname)
                header = ', '.join(('Data columns are: <z>', keyname, ' '))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
