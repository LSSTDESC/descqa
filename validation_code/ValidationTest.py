from __future__ import division, print_function
import os
from warnings import warn
import itertools
zip = itertools.izip

import numpy as np

import matplotlib
mpl = matplotlib
mpl.use('Agg') # Must be before importing matplotlib.pyplot
mpl.rcParams['font.size'] = 13.0
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.fontsize'] = 'x-small'
mpl.rcParams['figure.dpi'] = 200.0
mpl.rcParams['lines.markersize'] = 3.0
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.size'] = 5.0
mpl.rcParams['xtick.minor.size'] = 3.0
mpl.rcParams['ytick.major.size'] = 5.0
mpl.rcParams['ytick.minor.size'] = 3.0

import matplotlib.pyplot
plt = matplotlib.pyplot

import CalcStats

__all__ = ['ValidationTest', 'TestResult', 'mpl', 'plt', 'SimpleComparisonPlot', 'CalcStats']


class TestResult(object):
    """
    class for passing back test result
    """
    def __init__(self, score=None, summary='', passed=False, skipped=False, **kwargs):
        """
        Parameters
        ----------
        score : float or None
            a float number to represent the test score

        summary : str
            short summary string

        passed : bool
            if the test is passed

        skipped : bool
            if the test is skipped, overwrites all other arguments

        **kwargs :
            any other keyword arguments
        """

        self.skipped = bool(skipped)
        self.passed = bool(passed)
        self.summary = str(summary).strip()
        for k, v in kwargs:
            setattr(self, k, v)

        # set score
        if not self.skipped:
            try:
                self.score = float(score)
            except (TypeError, ValueError):
                if isinstance(score, basestring) and score.upper() in ('PASSED', 'FAILED', 'SKIPPED'):
                    # this is for backward compatibility in other validations
                    status = score.upper()
                    self.passed = (status == 'PASSED')
                    self.skipped = (status == 'SKIPPED')
                else:
                    raise ValueError('Must set a float value for `score`')


class ValidationTest(object):
    """
    abstract class for validation test class
    """

    _available_observations = {}
    _required_quantities = dict()
    _default_kwargs = dict()
    _plot_config = dict()
    _default_kwargs = {
        'zlo': 0.0,
        'zhi': 1000.0,
        'jackknife_nside': 5,
    }
    _output_filenames = dict(
        catalog_data='catalog_data.txt',
        validation_data='validation_data.txt',
        catalog_covariance='catalog_covariance.txt',
        logfile='logfile.txt',
        figure='figure.png',
    )

    def _import_kwargs(self, kwargs, key, attr_name=None, func=None, required=False, always_set=False):
        if attr_name is None:
            attr_name = '_{}'.format(key)
        val = kwargs.get(key, self._default_kwargs.get(key))
        if required and val is None:
            raise ValueError('Must specify test option `{}`'.format(key))
        if callable(func):
            val = func(val)
        if always_set or val is not None:
            setattr(self, attr_name, val)
        return val


    def __init__(self, **kwargs):
        self._import_kwargs(kwargs, 'base_data_dir', required=True)
        self._import_kwargs(kwargs, 'test_name')

        self._import_kwargs(kwargs, 'observation', always_set=True)
        if self._available_observations and self._observation not in self._available_observations:
            raise ValueError('`observation` not available')
        self._validation_name = self._observation

        self._import_kwargs(kwargs, 'bins', func=lambda b: np.logspace(*b), required=True)
        self._import_kwargs(kwargs, 'validation_range', always_set=True)
        self._import_kwargs(kwargs, 'jackknife_nside', func=int, required=True)
        self._import_kwargs(kwargs, 'zlo', func=float, required=True)
        self._import_kwargs(kwargs, 'zhi', func=float, required=True)
        self._zfilter = {'zlo': self._zlo, 'zhi': self._zhi}

        self._subclass_init(**kwargs)


    def _subclass_init(self, **kwargs):
        pass


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

        base_output_dir : string

        Returns
        -------
        test_result : TestResult object
            use the TestResult object to return test result
        """
        output_filenames = {k: os.path.join(base_output_dir, v) for k, v in self._output_filenames.iteritems()}

        #make sure galaxy catalog has appropriate quantities
        if not all(k in galaxy_catalog.quantities for k in self._required_quantities):
            #raise an informative warning
            msg = 'galaxy catalog {} does not have all the required quantities: {}, skipping the rest of the validation test.'.format(\
                    catalog_name, ', '.join(self._required_quantities))
            warn(msg)
            with open(output_filenames['logfile'], 'a') as f:
                f.write(msg)
            return TestResult(skipped=True)

        self._prepare_validation_test(galaxy_catalog, catalog_name, base_output_dir)
        catalog_result = self._calc_catalog_result(galaxy_catalog)

        self._save_data(output_filenames['validation_data'], self._validation_data)
        self._save_data(output_filenames['catalog_data'], catalog_result)
        if 'cov' in catalog_result:
            np.savetxt(output_filenames['catalog_covariance'], catalog_result['cov'])
        self._plot_result(output_filenames['figure'], catalog_result, catalog_name)

        return self._calculate_summary_statistic(catalog_result)


    def _calc_catalog_result(self, galaxy_catalog):
        raise NotImplementedError


    def _calculate_summary_statistic(self, catalog_result, passing_pvalue=0.95):
        if hasattr(self, '_interp_validation') and 'cov' not in self._validation_data:
            validation_data = self._interp_validation(catalog_result['x'])
        else:
            validation_data = self._validation_data

        #restrict range of validation data supplied for test if necessary
        if self._validation_range:
            mask_validation = (validation_data['x'] >= self._validation_range[0]) & (validation_data['x'] <= self._validation_range[1])
            mask_catalog = (catalog_result['x'] >= self._validation_range[0]) & (catalog_result['x'] <= self._validation_range[1])
        else:
            mask_validation = np.ones(len(validation_data['x']), dtype=bool)
            mask_catalog = np.ones(len(catalog_result['x']), dtype=bool)

        if np.count_nonzero(mask_validation) != np.count_nonzero(mask_catalog):
            raise ValueError('The length of validation data need to be the same as that of catalog result')

        d = validation_data['y'][mask_validation] - catalog_result['y'][mask_catalog]
        nbin = np.count_nonzero(mask_catalog)

        cov = np.zeros((nbin, nbin))
        if 'cov' in catalog_result:
            cov += catalog_result['cov'][np.outer(*(mask_catalog,)*2)].reshape(nbin, nbin)
        if 'cov' in validation_data:
            cov += validation_data['cov'][np.outer(*(mask_validation,)*2)].reshape(nbin, nbin)
        if not cov.any():
            raise ValueError('empty covariance')

        score, pvalue = CalcStats.chisq(d, cov, nbin)
        passed = (pvalue < passing_pvalue)
        msg = 'chi^2/dof = {:g}/{}; p-value = {:g} {} {:g}'.format(score, nbin, pvalue, '<' if passed else '>=', passing_pvalue)
        return TestResult(pvalue, msg, passed)


    def _plot_result(self, savepath, catalog_result, catalog_name, save_pdf=False):
        interp_validation = self._plot_config.get('plot_validation_as_line')
        with SimpleComparisonPlot(savepath, save_pdf) as plot:
            plot.plot_data(self._validation_data, self._validation_name, catalog_result, catalog_name, interp_validation, interp_validation)
            if self._validation_range:
                plot.add_vband(*self._validation_range)
            plot.set_labels(self._plot_config.get('xlabel'), self._plot_config.get('ylabel'), self._plot_config.get('ylabel_lower'), self._plot_config.get('title'))
            plot.set_lims(self._plot_config.get('xlim'), self._plot_config.get('ylim'), self._plot_config.get('ylim_lower'))
            plot.add_legend()


    @staticmethod
    def _save_data(filename, result, comment=''):
        fields = ('x', 'y', 'y-', 'y+') if 'y-' in result and 'y+' in result else ('x', 'y')
        np.savetxt(filename, np.vstack((result[k] for k in fields)).T, header=comment)


    @staticmethod
    def _load_data(filename):
        raw = np.loadtxt(filename).T
        fields = ('x', 'y', 'y-', 'y+') if len(raw) == 4 else ('x', 'y')
        return dict(zip(fields, raw))


    def plot_summary(self, output_file, catalog_list, save_pdf=True):
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
        data = []
        labels = []
        for catalog_name, catalog_dir in catalog_list:
            labels.append(catalog_name)
            data.append(self._load_data(os.path.join(catalog_dir, self._output_filenames['catalog_data'])))
        self._plot_result(output_file, data, labels, save_pdf)


_colors = ('#009292', '#ff6db6', '#490092', '#6db6ff', '#924900', '#24ff24')
_linestyles = ('-', '--', '-.', ':')

class SimpleComparisonPlot():
    def __init__(self, savefig_path=None, save_pdf=False, logx=True, logy=True):
        self.savefig_path = savefig_path
        self.save_pdf = save_pdf
        self.logx = logx
        self.logy = logy
        self.fig = None
        self.ax = None
        self.ax_lower = None


    def __enter__(self):
        self.fig, (self.ax, self.ax_lower) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': (1, 0.3), 'hspace':0})
        self.ax.set_xscale('log' if self.logx else 'linear')
        self.ax_lower.set_xscale('log' if self.logx else 'linear')
        self.ax.set_yscale('log' if self.logy else 'linear')
        self.ax_lower.set_yscale('linear')
        return self


    def __exit__(self, *exc_args):
        self.ax_lower.axhline(0.0, c='k', lw=0.5)
        self.ax_lower.minorticks_on()
        for t in self.ax_lower.yaxis.get_major_ticks()[-1:]:
            t.label1.set_visible(False)
        self.fig.tight_layout()
        if self.savefig_path:
            self.fig.savefig(self.savefig_path)
            if self.save_pdf:
                self.fig.savefig(self.savefig_path+'.pdf')
        plt.close(self.fig)


    def plot_data(self, ref_data, ref_label, other_data, other_labels, ref_as_line=False, interp=False):
        if isinstance(other_labels, basestring):
            ref_color = 'C1'
            other_format = [('-', 'C0')]
            other_data = [other_data]
            other_labels = [other_labels]
        else:
            ref_color = 'k'
            other_format = itertools.cycle(itertools.product(_linestyles, _colors))
            #other_colors = mpl.cm.get_cmap('viridis')(np.linspace(0, 1, len(other_data)))
            #other_linestyles = ['--', '-']*((len(other_data)+1)//2)

        for data, label, (ls, color) in zip(other_data, other_labels, other_format):
            self.add_line(self.mask_data(data), label, color, ls)
            self.add_line(self.compare_data(ref_data, data, interp), label, color, ls, lower=True)

        add_ref = self.add_line if ref_as_line else self.add_points
        add_ref(self.mask_data(ref_data), ref_label, ref_color)
        add_ref(self.compare_data(ref_data, ref_data), ref_label, ref_color, lower=True)


    def compare_data(self, ref_data, this_data, interp=False):
        d = dict()
        d['x'] = this_data['x']
        ref_y = ref_data['y']
        if interp:
            s = ref_data['x'].argsort()
            ref_x = ref_data['x'][s]
            this_x = d['x']
            if self.logx:
                ref_x = np.log(ref_x)
                this_x = np.log(this_x)
            ref_y = ref_data['y'][s]
            if self.logy:
                ref_y = np.log(ref_y)
            ref_y = np.interp(this_x, ref_x, ref_y)
            if self.logy:
                ref_y = np.exp(ref_y)
        for k in ('y', 'y+', 'y-'):
            if k in this_data:
                d[k] = this_data[k]/ref_y if self.logy else (this_data[k]-ref_y)
        d = self.mask_data(d)
        if self.logy:
            for k in ('y', 'y+', 'y-'):
                if k in this_data:
                    d[k] = np.log(d[k])
        return d


    def mask_data(self, data):
        if self.logy:
            mask = np.isfinite(data['y']) & (data['y'] > 0)
            d = {k: v[mask] for k, v in data.iteritems()}
            if 'y-' in d:
                d['y-'][d['y-'] <= 0] = 1.0e-100
            return d
        return data


    def add_line(self, data, label, color, linestyle='-', lower=False):
        ax_this = self.ax_lower if lower else self.ax
        ax_this.plot(data['x'], data['y'], label=label, color=color, ls=linestyle)
        if 'y-' in data and 'y+' in data:
            ax_this.fill_between(data['x'], data['y-'], data['y+'], alpha=0.15, color=color, lw=0)


    def add_points(self, data, label, color, lower=False):
        ax_this = self.ax_lower if lower else self.ax
        if 'y-' in data and 'y+' in data:
            ax_this.errorbar(data['x'], data['y'], [data['y']-data['y-'], data['y+']-data['y']], label=label, color=color, marker='s', ls='')
        else:
            ax_this.plot(data['x'], data['y'], label=label, color=color, marker='s', ls='')


    def add_vband(self, x0, x1):
        for ax_this in (self.ax, self.ax_lower):
            xlim_lo, xlim_hi = ax_this.get_xlim()
            if self.logx:
                xlim_lo /= 1000.0
                xlim_hi *= 1000.0
            else:
                xlim_lo -= 1000.0
                xlim_hi += 1000.0
            ax_this.axvspan(xlim_lo, x0, alpha=0.1, color='k', lw=0)
            ax_this.axvspan(x1, xlim_hi, alpha=0.1, color='k', lw=0)


    def add_legend(self, **kwargs):
        d = dict(ncol=2)
        d.update(kwargs)
        self.ax.legend(**d)


    def set_lims(self, xlim=None, ylim=None, ylim_lower=None):
        if xlim:
            self.ax.set_xlim(xlim)
            self.ax_lower.set_xlim(xlim)
        if ylim:
            self.ax.set_ylim(ylim)
        if ylim_lower is None:
            ylim_lower = (-0.7, 0.7)
        self.ax_lower.set_ylim(ylim_lower)


    def set_labels(self, xlabel=None, ylabel=None, ylabel_lower=None, title=None):
        if xlabel:
            self.ax_lower.set_xlabel(xlabel)
        if ylabel:
            self.ax.set_ylabel(ylabel)
        if ylabel_lower is None:
            ylabel_lower = 'ln(ratio)' if self.logy else 'diff.'
        self.ax_lower.set_ylabel(ylabel_lower)
        if title:
            self.ax.set_title(title)

