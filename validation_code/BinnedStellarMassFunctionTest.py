from __future__ import (division, print_function, absolute_import)
import os
from warnings import warn
import numpy as np

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ValidationTest import ValidationTest, TestResult
import CalcStats

__all__ = ['BinnedStellarMassFunctionTest', 'write_file', 'load_file', 'OnePointFunctionPlot', 'plot_summary']


class BinnedStellarMassFunctionTest(ValidationTest):
    """
    validation test class object to compute stellar mass function bins
    """

    output_config = dict(\
            catalog_output_file='catalog_smf.txt',
            validation_output_file='validation_smf.txt',
            covariance_matrix_file='covariance_smf.txt',
            log_file='log_smf.txt',
            plot_file='plot_smf.png',
            plot_title='Stellar Mass Function',
            xaxis_label=r'$\log (M^*/(M_\odot)$',
            yaxis_label=r'$dn/dV\,d\log M ({\rm Mpc}^{-3}\,{\rm dex}^{-1})$',
            plot_validation_as_line=False,
    )

    required_quantities = ('stellar_mass', 'positionX', 'positionY', 'positionZ')

    available_observations = ('LiWhite2009', 'MassiveBlackII')

    default_kwargs = {
            'observation': 'LiWhite2009',
            'zlo': 0,
            'zhi': 1000.0,
            'summary_statistic': 'chisq',
            'jackknife_nside': 5,
    }

    enable_interp_validation = None


    def _import_kwargs(self, kwargs, key, attr_name=None, func=None, required=False):
        if attr_name is None:
            attr_name = key
        val = kwargs.get(key, self.default_kwargs.get(key))
        if required and val is None:
            raise ValueError('Must specify test option `{}`'.format(key))
        if callable(func):
            val = func(val)
        setattr(self, attr_name, val)
        return val


    def __init__(self, **kwargs):
        """
        initialize a stellar mass function validation test

        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data

        base_output_dir : string
            base directory to store test data, e.g. plots

        test_name : string
            string indicating test name

        observation : string, optional
            name of stellar mass validation observation:
            LiWhite2009, MassiveBlackII

        bins : tuple, optional

        zlo : float, optional

        zhi : float, optional

        summary_details : boolean, optional

        summary_method : string, optional

        threshold : float, optional

        validation_range : tuple, optional
        """

        super(BinnedStellarMassFunctionTest, self).__init__(**kwargs)

        #load validation data
        self._import_kwargs(kwargs, 'observation')
        if self.available_observations and self.observation not in self.available_observations:
            raise ValueError('`observation` not available')

        self._import_kwargs(kwargs, 'bins', func=lambda b: np.linspace(*b), required=True)
        if self._import_kwargs(kwargs, 'validation_range') is None:
            self.validation_range = self.bins[[0, -1]]

        #redshift range
        self._import_kwargs(kwargs, 'zlo', func=float, required=True)
        self._import_kwargs(kwargs, 'zhi', func=float, required=True)
        self._zfilter = {'zlo': self.zlo, 'zhi': self.zhi}

        #statistic options
        self._import_kwargs(kwargs, 'summary_statistic', required=True)

        if self.summary_statistic == 'chisq':
            self._import_kwargs(kwargs, 'jackknife_nside', func=int, required=True)
        elif self.summary_statistic == 'lpnorm':
            self.jackknife_nside = 0
        else:
            raise ValueError('`summary_statistic` not available')

        #add all other options
        for k in kwargs:
            if not hasattr(self, k):
                setattr(self, k, kwargs[k])

        self._init_follow_up(kwargs)
        self.load_validation_data()


    def _init_follow_up(self, kwargs):
        pass


    def load_validation_data(self):
        """
        load tabulated stellar mass function data
        """

        #associate files with observations
        stellar_mass_function_files = {'LiWhite2009':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt',
                                       'MassiveBlackII':'LIWHITE/StellarMassFunction/massfunc_dataerr.txt'}

        #set the columns to use in each file
        columns = {'LiWhite2009':(0,5,6),
                   'MassiveBlackII':(0,1,2),}

        #get path to file
        fn = os.path.join(self.base_data_dir, stellar_mass_function_files[self.observation])

        #column 1: stellar mass bin center
        #column 2: number density
        #column 3: 1-sigma error
        binctr, mhist, merr = np.loadtxt(fn, unpack=True, usecols=columns[self.observation])

        self.validation_data = {'x':np.log10(binctr), 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':np.diag(merr*merr)}
        return self.validation_data


    def _prepare_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        pass


    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        run the validation test

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader

        catalog_name : string
            name of galaxy catalog

        Returns
        -------
        test_result : TestResult object
            use the TestResult object to return test result
        """

        #make sure galaxy catalog has appropriate quantities
        if not all(k in galaxy_catalog.quantities for k in self.required_quantities):
            #raise an informative warning
            msg = 'galaxy catalog {} does not have all the required quantities: {}, skipping the rest of the validation test.'.format(\
                    catalog_name, ', '.join(self.required_quantities))
            warn(msg)
            with open(os.path.join(base_output_dir, self.log_file), 'a') as f:
                f.write(msg)
            return TestResult(skipped=True)

        self._prepare_validation_test(galaxy_catalog, catalog_name, base_output_dir)

        #calculate stellar mass function in galaxy catalog
        catalog_result = self.calc_catalog_result(galaxy_catalog)

        #save results to files
        fn = os.path.join(base_output_dir, self.output_config['catalog_output_file'])
        write_file(catalog_result, fn)

        fn = os.path.join(base_output_dir, self.output_config['validation_output_file'])
        write_file(self.validation_data, fn)

        if 'cov' in catalog_result:
            fn = os.path.join(base_output_dir, self.output_config['covariance_matrix_file'])
            np.savetxt(fn, catalog_result['cov'])

        #plot results
        fn = os.path.join(base_output_dir, self.output_config['plot_file'])
        self.plot_result(catalog_result, catalog_name, fn)

        return self.calculate_summary_statistic(catalog_result)


    def get_quantities_from_catalog(self, galaxy_catalog):
        """
        obtain the masses and mask fom the galaxy catalog

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """
        #get stellar masses from galaxy catalog
        sm = galaxy_catalog.get_quantities("stellar_mass", self._zfilter)
        x = galaxy_catalog.get_quantities("positionX", self._zfilter)
        y = galaxy_catalog.get_quantities("positionY", self._zfilter)
        z = galaxy_catalog.get_quantities("positionZ", self._zfilter)

        #remove non-finite or negative numbers
        mask = np.isfinite(sm)
        mask &= (sm > 0)
        mask &= np.isfinite(x)
        mask &= np.isfinite(y)
        mask &= np.isfinite(z)

        return dict(mass=sm[mask], x=x[mask], y=y[mask], z=z[mask])


    def calc_catalog_result(self, galaxy_catalog):
        """
        calculate the stellar mass function in bins

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
        """

        #get stellar masses from galaxy catalog
        quantities = self.get_quantities_from_catalog(galaxy_catalog)
        log_mass = np.log10(quantities['mass'])

        #histogram points and compute bin positions
        mhist = np.histogram(log_mass, bins=self.bins)[0].astype(float)
        summass = np.histogram(log_mass, bins=self.bins, weights=quantities['mass'])[0]
        binwid = self.bins[1:] - self.bins[:-1]
        binctr = (self.bins[1:] + self.bins[:-1])*0.5
        has_mass = (mhist > 0)
        binctr[has_mass] = np.log10(summass/mhist)[has_mass]

        #count galaxies in log bins
        #get errors from jackknife samples if requested
        if self.jackknife_nside > 0:
            jack_indices = CalcStats.get_subvolume_indices(quantities['x'], quantities['y'], quantities['z'], \
                    galaxy_catalog.box_size, self.jackknife_nside)
            njack = self.jackknife_nside**3
            mhist, _, covariance = CalcStats.jackknife(log_mass, jack_indices, njack, \
                    lambda m, scale: np.histogram(m, bins=self.bins)[0]*scale, \
                    full_args=(1.0,), jack_args=(njack/(njack-1.0),))
        else:
            covariance = np.diag(mhist)

        #calculate number differential density
        vol = galaxy_catalog.box_size**3.0
        mhist /= (binwid * vol)
        covariance /= (vol*vol)
        covariance /= np.outer(binwid, binwid)
        merr = np.sqrt(np.diag(covariance))

        return {'x':binctr, 'y':mhist, 'y-':mhist-merr, 'y+':mhist+merr, 'cov':covariance}


    def interp_validation(self, x):
        """
        interpolate validation data
        """
        res = {}
        s = self.validation_data['x'].argsort()
        x_orig = self.validation_data['x'][s]
        res['x'] = x
        for k in ('y', 'y-', 'y+'):
            if k in self.validation_data:
                res[k] = np.exp(np.interp(x, x_orig, np.log(self.validation_data[k][s])))
        return res


    def calculate_summary_statistic(self, catalog_result):
        """
        Run summary statistic.

        Parameters
        ----------
        catalog_result :

        Returns
        -------
        result :  numerical result

        test_passed : boolean
            True if the test is passed, False otherwise.
        """

        if self.enable_interp_validation and 'cov' not in self.validation_data:
            validation_data = self.interp_validation(catalog_result['x'])
        else:
            validation_data = self.validation_data

        #restrict range of validation data supplied for test if necessary
        mask_validation = (validation_data['x'] >= self.validation_range[0]) & (validation_data['x'] <= self.validation_range[1])
        mask_catalog = (catalog_result['x'] >= self.validation_range[0]) & (catalog_result['x'] <= self.validation_range[1])
        if np.count_nonzero(mask_validation) != np.count_nonzero(mask_catalog):
            raise ValueError('The length of validation data need to be the same as that of catalog result')

        d = validation_data['y'][mask_validation] - catalog_result['y'][mask_catalog]
        nbin = np.count_nonzero(mask_catalog)

        if self.summary_statistic == 'chisq':
            cov = np.zeros((nbin, nbin))
            if 'cov' in catalog_result:
                cov += catalog_result['cov'][np.outer(*(mask_catalog,)*2)].reshape(nbin, nbin)
            if 'cov' in validation_data:
                cov += validation_data['cov'][np.outer(*(mask_validation,)*2)].reshape(nbin, nbin)
            if not cov.any():
                raise ValueError('empty covariance')

            score = CalcStats.chisq(d, cov)
            conf_level = getattr(self, 'chisq_conf_level', 0.95)
            score_thres = CalcStats.chisq_threshold(nbin, conf_level)
            passed = score < score_thres
            msg = 'chi^2 = {:g} {} {:g} confidence level = {:g}'.format(score, '<' if passed else '>=', conf_level, score_thres)

        elif self.summary_statistic == 'lpnorm':
            p = getattr(self, 'lpnorm_p', 2.0)
            score = CalcStats.Lp_norm(d, p)
            score_thres = getattr(self, 'lpnorm_thres', 1.0)
            passed = score < score_thres
            msg = 'L{:g} norm = {:g} {} test threshold {:g}'.format(p, score, '<' if passed else '>=', score_thres)

        return TestResult(score, msg, passed)


    def plot_result(self, result, catalog_name, savepath):
        """
        plot the stellar mass function of the catalog and validation data

        Parameters
        ----------
        result : dictionary
            stellar mass function of galaxy catalog

        catalog_name : string
            name of galaxy catalog

        savepath : string
            file to save plot
        """
        config = self.output_config
        with OnePointFunctionPlot(savepath, title=config['plot_title'], xlabel=config['xaxis_label'], ylabel=config['yaxis_label']) as plot:
            plot.add_line(result, label=catalog_name, color='b')
            if config['plot_validation_as_line']:
                plot.add_line(self.validation_data, label=self.observation, color='g')
            else:
                plot.add_points(self.validation_data, label=self.observation, marker='o', color='g')
            plot.add_vband(*self.validation_range, color='r', label='Test Region')


    @classmethod
    def plot_summary(self, output_file, catalog_list, validation_kwargs):
        """
        make summary plot for validation test

        Parameters
        ----------
        output_file: string
            filename for summary plot

        catalog_list: list of tuple
            list of (catalog, catalog_output_dir) used for each catalog comparison

        validation_kwargs : dict
            keyword arguments used in the validation
        """

        config = self.output_config
        colors = matplotlib.cm.get_cmap('nipy_spectral')(np.linspace(0, 1, len(catalog_list)+1)[:-1])

        with OnePointFunctionPlot(output_file, title=config['plot_title'], xlabel=config['xaxis_label'], ylabel=config['yaxis_label']) as plot:
            for color, (catalog_name, catalog_dir) in zip(colors, catalog_list):
                d = load_file(os.path.join(catalog_dir, config['catalog_output_file']))
                plot.add_line(d, catalog_name, color=color)

            d = load_file(os.path.join(catalog_dir, config['validation_output_file']))
            if config['plot_validation_as_line']:
                plot.add_line(d, label=validation_kwargs['observation'], color='k')
            else:
                plot.add_points(d, validation_kwargs['observation'], color='k', marker='o')


plot_summary = BinnedStellarMassFunctionTest.plot_summary


def write_file(result, filename, comment=''):
    """
    write stellar mass function data file

    Parameters
    ----------
    result : dictionary

    filename : string

    comment : string
    """
    if 'y-' in result and 'y+' in result:
        fields = ('x', 'y', 'y-', 'y+')
    else:
        fields = ('x', 'y')
    np.savetxt(filename, np.vstack((result[k] for k in fields)).T,
                fmt='%13.6e', header=comment)


def load_file(filename):
    """
    write stellar mass function data file

    Parameters
    ----------
    result : dictionary

    filename : string

    comment : string
    """
    raw = np.loadtxt(filename).T
    fields = ('x', 'y', 'y-', 'y+') if len(raw) == 4 else ('x', 'y')
    return dict(zip(fields, raw))


class OnePointFunctionPlot():
    def __init__(self, savefig, **kwargs):
        self.savefig = savefig
        self.kwargs = kwargs

    def __enter__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xscale('linear')
        self.ax.set_yscale('log')
        return self

    def __exit__(self, *exc_args):
        self.ax.set_xlabel(self.kwargs['xlabel'])
        self.ax.set_ylabel(self.kwargs['ylabel'])
        self.ax.set_title(self.kwargs['title'])
        self.ax.legend(loc='best', frameon=False, fontsize='small', ncol=2)
        self.fig.tight_layout()
        self.fig.savefig(self.savefig)
        plt.close(self.fig)

    def add_line(self, d, label, **kwargs):
        mask = (d['y'] > 0)
        l = self.ax.plot(d['x'][mask], d['y'][mask], label=label, lw=1.5, **kwargs)[0]
        if 'y-' in d and 'y+' in d:
            self.ax.fill_between(d['x'][mask], d['y-'][mask], d['y+'][mask], alpha=0.2, color=l.get_color(), lw=0)

    def add_points(self, d, label, **kwargs):
        if 'y-' in d and 'y+' in d:
            self.ax.errorbar(d['x'], d['y'], [d['y']-d['y-'], d['y+']-d['y']], label=label, ls='', **kwargs)
        else:
            self.ax.plot(d['x'], d['y'], 'o', label=label, **kwargs)

    def add_vband(self, x0, x1, **kwargs):
        ymin, ymax = self.ax.get_ylim()
        plt.fill_between([x0, x1], [ymin, ymin], [ymax, ymax], alpha=0.15, **kwargs)



