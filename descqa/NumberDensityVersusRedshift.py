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

__all__ = ['NumberDensityVersusRedshift']


class NumberDensityVersusRedshift(BaseValidationTest):
    """
    validation test to show N(z) distributions
    """
    #setup dict with parameters needed to read in validation data
    possible_observations = {
        'Coil2004_magbin': {
            'filename_template': 'N_z/DEEP2/Coil_et_al_2004_Table3_{}.txt',
            'usecols': (0, 1, 2, 4),
            'colnames': ('mag_hi', 'mag_lo', 'z0values', 'z0errors'),
            'skiprows': 2,
            'label': 'Coil et. al. 2004',
        },
        'Coil2004_maglim': {
            'filename_template': 'N_z/DEEP2/Coil_et_al_2004_Table4_{}.txt',
            'usecols': (0, 1, 2),
            'colnames': ('mag_hi', 'mag_lo', 'z0values'),
            'skiprows': 3,
            'label': 'Coil et. al. 2004',
        },
        'DEEP2_JAN': {
            'filename_template': 'N_z/DEEP2/JANewman_{}.txt',
            'usecols': (0, 1, 2, 3),
            'colnames': ('mag_hi_lim', 'mag_lo_lim', 'z0const', 'z0linear'),
            'skiprows': 1,
            'label': 'DEEP2 (JAN, p.c.)',
        },
    }

    #plotting constants
    figx_p = 9
    figy_p = 11
    lw2 = 2
    fsize = 16 #fontsize
    lsize = 10 #legendsize
    msize = 6  #markersize
    default_colors = ['blue', 'r', 'm', 'g', 'navy', 'y', 'purple', 'gray', 'c',\
        'orange', 'violet', 'coral', 'gold', 'orchid', 'maroon', 'tomato', \
        'sienna', 'chartreuse', 'firebrick', 'SteelBlue']
    validation_color = 'black'
    default_markers = ['o', 'v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']

    def __init__(self, z='redshift_true', band='i', N_zbins=44, zlo=0., zhi=1.1,
                 observation='', mag_lo=27, mag_hi=18, ncolumns=2, normed=True,
                 **kwargs): #pylint: disable=W0231

        #catalog quantities
        self.zlabel = z
        possible_mag_fields = ('mag_{}_lsst',
                               'mag_{}_sdss',
                               'mag_{}_des',
                               'mag_true_{}_lsst',
                               'mag_true_{}_sdss',
                               'mag_true_{}_des',
                              )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]
        self.band = band

        #z-bounds and binning
        self.zlo = zlo
        self.zhi = zhi
        self.N_zbins = N_zbins
        self.zbins = np.linspace(zlo, zhi, N_zbins+1)
        self.filters = [(lambda z: (z > zlo) & (z < zhi), self.zlabel)]

        #validation data
        self.validation_data = {}
        self.observation = observation

        #check for valid observations
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown')
        elif observation not in self.possible_observations:
            raise ValueError('Observation {} not available'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)

        #plotting variables
        self.normed = normed
        self.ncolumns = int(ncolumns)

        #setup subplot configuration and get magnitude cuts for each plot
        self.mag_lo, self.mag_hi = self.init_plots(mag_lo, mag_hi)

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p), sharex='col')
        #could plot summary validation data here if available but would need to evaluate labels, bin values etc.
        #otherwise setup a check so that validation data is plotted only once on summary plot
        self.first_pass = True

        self._other_kwargs = kwargs


    def init_plots(self, mlo, mhi):
        #get magnitude cuts based on validation data or default limits (only catalog data plotted)
        mag_lo = self.validation_data.get('mag_lo', [float(m) for m in range(int(mhi), int(mlo+1))])
        mag_hi = self.validation_data.get('mag_hi', [])

        #setup number of plots and number of rows required for subplots
        self.nplots = len(mag_lo)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns

        #other plotting variables
        self.colors = iter(self.default_colors)
        self.markers = iter(self.default_markers)
        self.yaxis = 'P(z|m)' if self.normed else 'N(z|m)'

        return mag_lo, mag_hi


    def get_validation_data(self, band, observation):
        data_args = self.possible_observations[observation]
        data_path = os.path.join(self.data_dir, data_args['filename_template'].format(band))

        if not os.path.exists(data_path):
            raise ValueError("{}-band data file {} not found".format(band, data_path))

        if not os.path.getsize(data_path):
            raise ValueError("{}-band data file {} is empty".format(band, data_path))

        data = np.loadtxt(data_path, unpack=True, usecols=data_args['usecols'], skiprows=data_args['skiprows'])

        validation_data = dict(zip(data_args['colnames'], data))
        validation_data['label'] = data_args['label']

        #set mag_lo and mag_hi for cases where range of magnitudes is given
        if 'mag_lo' not in validation_data:
            validation_data['mag_hi'] = []
            validation_data['mag_lo'] = [float(m) for m in range(int(validation_data['mag_hi_lim']), int(validation_data['mag_lo_lim'])+1)]

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #check catalog data for required quantities
        mag_field = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped=True, summary='Missing required mag_field option')
        if not catalog_instance.has_quantity(self.zlabel):
            return TestResult(skipped=True, summary='Missing required {} quantity'.format(self.zlabel))
        filtername = mag_field.rpartition('_')[-1].upper()
        filelabel = '_'.join((filtername, self.band))

        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p), sharex='col')
        catalog_color = next(self.colors)
        catalog_marker= next(self.markers)

        #initialize arrays for storing histogram sums
        N_array = np.zeros((self.nrows, self.ncolumns, len(self.zbins)-1), dtype=np.int)
        sumz_array = np.zeros((self.nrows, self.ncolumns,len(self.zbins)-1))

        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        for catalog_data in catalog_instance.get_quantities([self.zlabel, mag_field], filters=self.filters, return_iterator=True):
            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)
            for cut_lo, cut_hi, N, sumz in zip_longest(
                    self.mag_lo,
                    self.mag_hi,
                    N_array.reshape(-1, N_array.shape[-1]), #flatten all but last dimension of array
                    sumz_array.reshape(-1, sumz_array.shape[-1]),
            ):
                if cut_lo:
                    mask = (catalog_data[mag_field] < cut_lo)
                    if cut_hi:
                        mask &= (catalog_data[mag_field] >= cut_hi)
                    z_this = catalog_data[self.zlabel][mask]
                    del mask

                    #bin catalog_data and accumulate subplot histograms
                    N += np.histogram(z_this, bins=self.zbins)[0]
                    sumz += np.histogram(z_this, bins=self.zbins, weights=z_this)[0]

        #loop over magnitude cuts and make plots
        results = {}
        for n, (ax_this, summary_ax_this, cut_lo, cut_hi, N, sumz, z0, z0err) in enumerate(zip_longest(
                ax.flat,
                self.summary_ax.flat,
                self.mag_lo,
                self.mag_hi,
                N_array.reshape(-1, N_array.shape[-1]),
                sumz_array.reshape(-1, sumz_array.shape[-1]),
                self.validation_data.get('z0values', []),
                self.validation_data.get('z0errors', []),
        )):
            if cut_lo is None:  #cut_lo is None if self.mag_lo is exhausted
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)
            else:
                cut_label = '{} $< {}$'.format(self.band, cut_lo)
                if cut_hi:
                    cut_label = '${} <=$ '.format(cut_hi) + cut_label #also appears in txt file so don't use \leq

                if z0 is None and 'z0const' in self.validation_data:  #alternate format for some validation data
                    z0 = self.validation_data['z0const'] + self.validation_data['z0linear'] * cut_lo

                meanz = sumz/N
                sumN = N.sum()
                total = '(# of galaxies = {})'.format(sumN)
                Nerrors = np.sqrt(N)
                if self.normed:
                    binwidths = self.zbins[1:] - self.zbins[:-1]
                    N = N/sumN/binwidths
                    Nerrors = Nerrors/sumN/binwidths

                #make subplot
                catalog_label = ' '.join((catalog_name, cut_label.replace(self.band, filtername + ' ' + self.band)))
                validation_label = ' '.join((self.validation_data.get('label', ''), cut_label))
                key = cut_label.replace('$', '')
                results[key] = {'meanz': meanz, 'total':total, 'N':N, 'N+-':Nerrors}
                self.catalog_subplot(ax_this, meanz, N, Nerrors, catalog_color, catalog_marker, catalog_label)
                if z0 and z0 > 0.:
                    fits = self.validation_subplot(ax_this, meanz, z0, z0err, validation_label)
                    results[key].update(fits)
                self.decorate_subplot(ax_this, n)

                #add curve for this catalog to summary plot
                self.catalog_subplot(summary_ax_this, meanz, N, Nerrors, catalog_color, catalog_marker, catalog_label)
                if self.first_pass and z0 and z0 > 0:
                    self.validation_subplot(summary_ax_this, meanz, z0, z0err, validation_label) #add validation data if evaluating first catalog
                self.decorate_subplot(summary_ax_this, n)


        #save results for catalog and validation data in txt files
        for filename, dtype, comment, info in zip_longest((filelabel, self.observation), ('N', 'fit'), (filtername,), ('total',)):
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
        return TestResult(inspect_only=True)


    def catalog_subplot(self, ax, meanz, data, errors, catalog_color, catalog_marker, catalog_label):

        ax.errorbar(meanz, data, yerr=errors, label=catalog_label, color=catalog_color, fmt=catalog_marker, ms=self.msize)


    def validation_subplot(self, ax, meanz, z0, z0err, validation_label):
        #plot validation data if available
        if not self.normed:
            raise ValueError("Only fits to normed plots are implemented so far")

        ndata = meanz**2*np.exp(-meanz/z0)
        norm = self.nz_norm(self.zhi, z0) - self.nz_norm(self.zlo, z0)
        ax.plot(meanz, ndata/norm, label=validation_label, ls='--', color=self.validation_color, lw=self.lw2)
        fits = {'fit': ndata/norm}

        if z0err and z0err > 0:
            nlo = meanz**2*np.exp(-meanz/(z0-z0err))
            nhi = meanz**2*np.exp(-meanz/(z0+z0err))
            normlo = self.nz_norm(self.zhi, z0-z0err) - self.nz_norm(self.zlo, z0-z0err)
            normhi = self.nz_norm(self.zhi, z0+z0err) - self.nz_norm(self.zlo, z0+z0err)
            ax.fill_between(meanz, nlo/normlo, nhi/normhi, alpha=0.3, facecolor=self.validation_color)
            fits['fit+'] = nhi/normhi
            fits['fit-'] = nlo/normlo

        return fits


    def decorate_subplot(self, ax, nplot):
        #add axes and legend
        if nplot % self.ncolumns == 0:  #1st column
            ax.set_ylabel('$'+self.yaxis+'$', size=self.fsize)

        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            #print "noticks",nplot
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(False)
                #prevent overlapping yaxis labels
                ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            ax.set_xlabel('$z$', size=self.fsize)
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(True)
        ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=self.lsize, numpoints=1)


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
            elif keyname+'+-' in results:
                fields = ('meanz', keyname, keyname+'+-')
                header = ', '.join(('Data columns are: <z>', keyname, keyname+'+-',' '))
            else:
                fields = ('meanz', keyname)
                header = ', '.join(('Data columns are: <z>', keyname, ' '))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
