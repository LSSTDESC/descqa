from __future__ import print_function, division, unicode_literals, absolute_import
import os
import math
import re
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest

import numpy as np
from GCR import GCRQuery
from sklearn.cluster import k_means

from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['NumberDensityVersusRedshift']


class NumberDensityVersusRedshift(BaseValidationTest):
    """
    Validation test to show redshift distribution P(z) or N(z)

    Parameters
    ----------
    z : str, optional, (default: 'redshift_true')
        label for redshift column
    band : str, optional (default: 'i')
        band to test
    N_zbins : int, optional (default: 10)
        number of redshift bins between `zlo` and `zhi`
        should be smaller than `N_jack` if `jackknife` is set to `True`
    zlo : float, optional, (default: 0)
        lower redshift limit
    zhi : float, optional (default: 1.1)
        upper redshift limit
    observation : str, optional (default: '')
        observation dataset to compare to
    mag_lo : float, optional (default: 27)
        faint-end magnitude limit
    mag_hi : float, optional (default: 18)
        bright-end magnitude limit
    ncolumns : int, optional (default: 2)
        number of subplot columns
    normed : bool, optional (default: True)
        normalize the redshift distribution (i.e. plotting P(z)).
        Note that when `normed` set to `False` the comparision with validation data
        does not make much sense since the validation data is normalized.
    jackknife : bool, optional (default: False)
        turn on jackknife error. When set to `False` use Poisson error.
    N_jack : int, optional (default: 20)
        number of jackknife regions
        `N_jack` should be much larger than `N_zbins` for the jackknife errors to be stable
    ra : str, optional, (default: 'ra')
        label of RA column (used if `jackknife` is `True`)
    dec : str, optional, (default: 'dec')
        label of Dec column (used if `jackknife` is `True`)
    pass_limit : float, optional (default: 2.)
        chi^2 value needs to be less than this value to pass the test
    use_diagonal_only : bool, optional (default: False)
        use only the diagonal terms of the convariance matric when calculating chi^2
    rest_frame: boolean, optional (default: False)
        use rest-frame magnitudes for cuts
        Note that mag_lo and mag_hi need to be adjusted if rest_frame is set to `True`
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
            'label': 'DEEP2',
        },
    }

    #plotting constants
    figx_p = 9
    figy_p = 11
    lw2 = 2
    msize = 6  #markersize
    default_colors = ['blue', 'r', 'm', 'g', 'navy', 'y', 'purple', 'gray', 'c',\
        'orange', 'violet', 'coral', 'gold', 'orchid', 'maroon', 'tomato', \
        'sienna', 'chartreuse', 'firebrick', 'SteelBlue']
    validation_color = 'black'
    default_markers = ['o', 'v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']

    def __init__(self, band='i', N_zbins=10, zlo=0., zhi=1.1,
                 observation='', mag_lo=27, mag_hi=18, ncolumns=2, normed=True,
                 jackknife=False, N_jack=20, ra='ra', dec='dec', pass_limit=2.,
                 use_diagonal_only=False, rest_frame=False, **kwargs):
        # pylint: disable=W0231

        #catalog quantities
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.replace_cat_name = kwargs.get('replace_cat_name', {})
        self.title_in_legend = kwargs.get('title_in_legend', False)
        self.legend_location = kwargs.get('legend_location', 'upper left')
        self.font_size = kwargs.get('font_size', 16)
        self.legend_size = kwargs.get('legend_size', 10)
        self.tick_size = kwargs.get('tick_size', 12)
        self.rest_frame = rest_frame
        if self.rest_frame:
            possible_mag_fields = ('Mag_true_{}_lsst_z0',
                                   'Mag_true_{}_sdss_z0',
                                   'Mag_true_{}_des_z0',
                                  )
        else:
            possible_mag_fields = ('mag_{}_cModel',
                                   'mag_{}_lsst',
                                   'mag_{}_sdss',
                                   'mag_{}_des',
                                   'mag_true_{}_lsst',
                                   'mag_true_{}_sdss',
                                   'mag_true_{}_des',
                                  )
        self.possible_mag_fields = [f.format(band) for f in possible_mag_fields]
        self.possible_redshifts = ['redshift_true_galaxy', 'redshift_true']
        self.band = band

        #z-bounds and binning
        self.zlo = zlo
        self.zhi = zhi
        self.N_zbins = N_zbins
        self.zbins = np.linspace(zlo, zhi, N_zbins+1)

        #errors
        self.jackknife = jackknife
        self.N_jack = N_jack
        self.ra = ra
        self.dec = dec
        self.use_diagonal_only = use_diagonal_only

        #scores
        self.pass_limit = pass_limit

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
        print(mag_lo, mag_hi)
        #check if supplied limits differ from validation limits and adjust
        mask = (mag_lo <= float(mlo)) & (mag_lo >= float(mhi))
        if np.count_nonzero(mask) < len(mag_lo):
            if len(mag_hi) > 0:
                mag_hi = mag_hi[mask]
                self.validation_data['mag_hi'] = mag_hi
            mag_lo = mag_lo[mask]
            self.validation_data['mag_lo'] = mag_lo
            if 'z0values' in self.validation_data:
                self.validation_data['z0values'] = self.validation_data['z0values'][mask]
            if  'z0errors' in self.validation_data:
                self.validation_data['z0errors'] = self.validation_data['z0errors'][mask]

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
            validation_data['mag_lo'] = np.asarray([float(m) for m in range(int(validation_data['mag_hi_lim']),
                                                                            int(validation_data['mag_lo_lim'])+1)])

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #check catalog data for required quantities
        mag_field = catalog_instance.first_available(*self.possible_mag_fields)
        if not mag_field:
            return TestResult(skipped=True, summary='Missing required mag_field option')
        self.zlabel = catalog_instance.first_available(*self.possible_redshifts)
        if not self.zlabel:
            return TestResult(skipped=True, summary='Missing required redhsift option')
        self.filters = [(lambda z: (z > self.zlo) & (z < self.zhi), self.zlabel)]

        jackknife_quantities = [self.zlabel, self.ra, self.dec] if self.jackknife else [self.zlabel]
        for jq in jackknife_quantities:
            if not catalog_instance.has_quantity(jq):
                return TestResult(skipped=True, summary='Missing required {} quantity'.format(jq))

        required_quantities = jackknife_quantities + [mag_field]
        filtername = mag_field.split('_')[(-1 if mag_field.startswith('m') else -2)].upper()  #extract filtername
        filelabel = '_'.join((filtername, self.band))

        #setup plots
        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]
        if self.replace_cat_name:
            for k, v in self.replace_cat_name.items():
                catalog_name = re.sub(k, v, catalog_name)
                                                                                    
        fig, ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p), sharex='col')
        catalog_color = next(self.colors)
        catalog_marker = next(self.markers)

        #initialize arrays for storing histogram sums
        N_array = np.zeros((self.nrows, self.ncolumns, len(self.zbins)-1), dtype=np.int)
        sumz_array = np.zeros((self.nrows, self.ncolumns, len(self.zbins)-1))

        jackknife_data = {}
        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        for catalog_data in catalog_instance.get_quantities(required_quantities, filters=self.filters, return_iterator=True):
            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)
            # filter catalog data further for matched object catalogs
            if np.ma.isMaskedArray(catalog_data[self.zlabel]):
                galmask = np.ma.getmask(catalog_data[self.zlabel])
                catalog_data = {k: v[galmask] for k, v in catalog_data.items()}

            for n, (cut_lo, cut_hi, N, sumz) in enumerate(zip_longest(
                    self.mag_lo,
                    self.mag_hi,
                    N_array.reshape(-1, N_array.shape[-1]), #flatten all but last dimension of array
                    sumz_array.reshape(-1, sumz_array.shape[-1]),
            )):
                if cut_lo:
                    mask = (catalog_data[mag_field] < cut_lo)
                    if cut_hi:
                        mask &= (catalog_data[mag_field] >= cut_hi)
                    z_this = catalog_data[self.zlabel][mask]

                    #save data for jackknife errors
                    if self.jackknife:   #store all the jackknife data in numpy arrays for later processing
                        if str(n) not in jackknife_data.keys(): #initialize sub-dict
                            jackknife_data[str(n)] = dict(zip(required_quantities, [np.asarray([]) for jq in jackknife_quantities]))
                        for jkey in jackknife_data[str(n)].keys():
                            jackknife_data[str(n)][jkey] = np.hstack((jackknife_data[str(n)][jkey], catalog_data[jkey][mask]))

                    del mask

                    #bin catalog_data and accumulate subplot histograms
                    N += np.histogram(z_this, bins=self.zbins)[0]
                    sumz += np.histogram(z_this, bins=self.zbins, weights=z_this)[0]


        #loop over magnitude cuts and make plots
        results = {}
        scores = np.array([self.pass_limit]*self.nplots)
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
                if ax_this is not None:
                    ax_this.set_visible(False)
                if summary_ax_this is not None:
                    summary_ax_this.set_visible(False)
            else:
                cut_label = '{} $< {}$'.format(self.band, cut_lo)
                if cut_hi:
                    cut_label = '${} \\leq $ {}'.format(cut_hi, cut_label) #also appears in txt file

                if z0 is None and 'z0const' in self.validation_data:  #alternate format for some validation data
                    z0 = self.validation_data['z0const'] + self.validation_data['z0linear'] * cut_lo

                N = N.astype(np.float64)

                if self.jackknife:
                    covariance = self.get_jackknife_errors(self.N_jack, jackknife_data[str(n)], N)
                else:
                    covariance = np.diag(N)

                meanz = sumz / N
                sumN = N.sum()
                total = '(# of galaxies = {})'.format(sumN)

                if self.normed:
                    scale = sumN * (self.zbins[1:] - self.zbins[:-1])
                    N /= scale
                    covariance /= np.outer(scale, scale)

                Nerrors = np.sqrt(np.diag(covariance))

                #make subplot
                catalog_label = ' '.join((catalog_name, cut_label.replace(self.band, filtername + ' ' + self.band)))
                validation_label = ' '.join((self.validation_data.get('label', ''), cut_label))
                key = cut_label.replace('$', '').replace('\\leq', '<=')
                results[key] = {'meanz': meanz, 'total':total, 'N':N, 'N+-':Nerrors}
                self.catalog_subplot(ax_this, meanz, N, Nerrors, catalog_color, catalog_marker, catalog_label)
                if z0 and z0 > 0: # has validation data
                    fits = self.validation_subplot(ax_this, meanz, z0, z0err, validation_label)
                    results[key].update(fits)
                    scores[n], inverse_cov = self.get_score(N, fits['fit'], covariance, use_diagonal_only=self.use_diagonal_only)
                    results[key]['score'] = 'Chi_sq/dof = {:11.4g}'.format(scores[n])
                    if self.jackknife:
                        results[key]['inverse_cov_matrix'] = inverse_cov

                self.decorate_subplot(ax_this, n)

                #add curve for this catalog to summary plot
                self.catalog_subplot(summary_ax_this, meanz, N, Nerrors, catalog_color, catalog_marker, catalog_label)
                if self.first_pass and z0 and z0 > 0:
                    self.validation_subplot(summary_ax_this, meanz, z0, z0err, validation_label) #add validation data if evaluating first catalog
                self.decorate_subplot(summary_ax_this, n)

        #save results for catalog and validation data in txt files
        for filename, dtype, comment, info, info2 in zip_longest((filelabel, self.observation), ('N', 'fit'),
                                                                 (filtername,), ('total', 'z0'), ('score', 'z0err')):
            if filename:
                with open(os.path.join(output_dir, 'Nvsz_' + filename + '.txt'), 'ab') as f_handle: #open file in append binary mode
                    #loop over magnitude cuts in results dict
                    for key, value in results.items():
                        self.save_quantities(dtype, value, f_handle, comment=' '.join(((comment or ''),
                                                                                       key, value.get(info, ''), value.get(info2, ''))))

                if self.jackknife:
                    with open(os.path.join(output_dir, 'Nvsz_' + filename + '.txt'), 'a') as f_handle: #open file in append mode
                        f_handle.write('\nInverse Covariance Matrices:\n')
                        for key in results.keys():
                            self.save_matrix(results[key]['inverse_cov_matrix'], f_handle, comment=key)


        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, 'Nvsz_' + filelabel + '.png'))
        plt.close(fig)

        #compute final score
        #final_scores = (scores < self.pass_limit)
        #pass or fail on average score rather than demanding that all distributions pass
        score_ave = np.mean(scores)
        return TestResult(score_ave, passed=score_ave < self.pass_limit)


    def get_jackknife_errors(self, N_jack, jackknife_data, N):
        nn = np.stack((jackknife_data[self.ra], jackknife_data[self.dec]), axis=1)
        _, jack_labels, _ = k_means(n_clusters=N_jack, random_state=0, X=nn)

        #make histograms for jackknife regions
        Njack_array = np.zeros((N_jack, len(self.zbins)-1), dtype=np.int)
        for nj in range(N_jack):
            Njack_array[nj] = np.histogram(jackknife_data[self.zlabel][jack_labels != nj], self.zbins)[0]

        covariance = np.zeros((self.N_zbins, self.N_zbins))
        for i in range(self.N_zbins):
            for j in range(self.N_zbins):
                for njack in Njack_array:
                    covariance[i][j] += (N_jack - 1.)/N_jack * (N[i] - njack[i]) * (N[j] - njack[j])

        return covariance


    def catalog_subplot(self, ax, meanz, data, errors, catalog_color, catalog_marker, catalog_label):
        ax.errorbar(meanz, data, yerr=errors, label=catalog_label, color=catalog_color, fmt=catalog_marker, ms=self.msize)


    def validation_subplot(self, ax, meanz, z0, z0err, validation_label):
        #plot validation data if available
        ndata = meanz**2*np.exp(-meanz/z0)
        norm = self.nz_norm(self.zhi, z0) - self.nz_norm(self.zlo, z0)
        ax.plot(meanz, ndata/norm, label=validation_label, ls='--', color=self.validation_color, lw=self.lw2)
        fits = {'fit': ndata/norm, 'z0':'z0 = {:.3f}'.format(z0)}

        if z0err and z0err > 0:
            nlo = meanz**2*np.exp(-meanz/(z0-z0err))
            nhi = meanz**2*np.exp(-meanz/(z0+z0err))
            normlo = self.nz_norm(self.zhi, z0-z0err) - self.nz_norm(self.zlo, z0-z0err)
            normhi = self.nz_norm(self.zhi, z0+z0err) - self.nz_norm(self.zlo, z0+z0err)
            ax.fill_between(meanz, nlo/normlo, nhi/normhi, alpha=0.3, facecolor=self.validation_color)
            fits['fit+'] = nhi/normhi
            fits['fit-'] = nlo/normlo
            fits['z0err'] = 'z0err = {:.3f}'.format(z0err)

        return fits


    def decorate_subplot(self, ax, nplot):
        #add axes and legend
        ax.tick_params(labelsize=self.tick_size)
        if nplot % self.ncolumns == 0:  #1st column
            ax.set_ylabel(self.yaxis, size=self.font_size)

        if nplot+1 <= self.nplots-self.ncolumns:  #x scales for last ncol plots only
            #print "noticks",nplot
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(False)
                #prevent overlapping yaxis labels
                ax.yaxis.get_major_ticks()[0].label1.set_visible(False)
        else:
            ax.set_xlabel('z', size=self.font_size)
            ax.tick_params(labelbottom=True)
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(True)
        ax.legend(loc='best', fancybox=True, framealpha=0.5, fontsize=self.legend_size, numpoints=1)


    @staticmethod
    def get_score(catalog, validation, cov, use_diagonal_only=True):

        #remove bad values
        mask = np.isfinite(catalog) & np.isfinite(validation)
        if not mask.any():
            return np.nan

        catalog = catalog[mask]
        validation = validation[mask]
        cov = cov[mask][:, mask]

        inverse_cov = np.diag(1.0 / np.diag(cov))
        if not use_diagonal_only:
            try:
                inverse_cov = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                print('Covariance matrix inversion failed: diagonal errors only will be used')

        d = catalog - validation
        chi2 = np.einsum('i,ij,j', d, inverse_cov, d)
        chi2_reduced = chi2 / float(len(catalog))
        return chi2_reduced, inverse_cov


    @staticmethod
    def nz_norm(z, z0):
        return z0*math.exp(-z/z0)*(-z*z-2.*z*z0-2.*z0*z0)


    @staticmethod
    def post_process_plot(fig):
        fig.subplots_adjust(hspace=0)


    @staticmethod
    def save_matrix(matrix, fhandle, comment=''):
        fhandle.write('{}:\n'.format(comment))
        for row in matrix:
            fhandle.write('  '.join(['{:10.3g}'.format(element) for element in row])+'\n')


    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('meanz', keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <z>', keyname, keyname+'-', keyname+'+', ' '))
            elif keyname+'+-' in results:
                fields = ('meanz', keyname, keyname+'+-')
                header = ', '.join(('Data columns are: <z>', keyname, keyname+'+-', ' '))
            else:
                fields = ('meanz', keyname)
                header = ', '.join(('Data columns are: <z>', keyname, ' '))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=header+comment)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
