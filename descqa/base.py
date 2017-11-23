from __future__ import division, unicode_literals, absolute_import
import os
from warnings import warn
import itertools
from builtins import zip, str

import numpy as np

from .plotting import *
from . import utils

__all__ = ['BaseValidationTest', 'ValidationTest', 'TestResult']


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
                if isinstance(score, str) and score.upper() in ('PASSED', 'FAILED', 'SKIPPED'):
                    # this is for backward compatibility in other validations
                    status = score.upper()
                    self.passed = (status == 'PASSED')
                    self.skipped = (status == 'SKIPPED')
                else:
                    raise ValueError('Must set a float value for `score`')


class BaseValidationTest(object):
    """
    very abstract class for validation test class
    """
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
        pass

    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        raise NotImplementedError

    def plot_summary(self, output_file, catalog_list, save_pdf=True):
        pass



class ValidationTest(BaseValidationTest):
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

    def __init__(self, **kwargs):
        self._import_kwargs(kwargs, 'base_data_dir', required=True)
        self._import_kwargs(kwargs, 'test_name')

        self._import_kwargs(kwargs, 'observation', always_set=True)
        if self._available_observations and self._observation not in self._available_observations:
            raise ValueError('`observation` not available')
        self._validation_name = self._observation

        self._import_kwargs(kwargs, 'bins', func=lambda b: np.logspace(*b), required=True)
        self._import_kwargs(kwargs, 'validation_range', func=lambda x: list(map(float, x)), always_set=True)
        self._import_kwargs(kwargs, 'jackknife_nside', func=int, required=True)
        self._import_kwargs(kwargs, 'zlo', func=float, required=True)
        self._import_kwargs(kwargs, 'zhi', func=float, required=True)
        self._zfilter = (lambda z: (z >= self._zlo) & (z < self._zhi), 'redshift_true')

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
        output_filenames = {k: os.path.join(base_output_dir, v) for k, v in self._output_filenames.items()}

        #make sure galaxy catalog has appropriate quantities
        if not galaxy_catalog.has_quantities(self._required_quantities):
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

        score, pvalue = utils.chisq(d, cov, nbin)
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
