from __future__ import print_function, division, unicode_literals, absolute_import
import os
from collections import defaultdict
import numpy as np
import scipy.special as scsp
import treecorr
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import (generate_uniform_random_ra_dec_footprint,
                   get_healpixel_footprint,
                   generate_uniform_random_dist)

__all__ = ['CorrelationsAngularTwoPoint', 'CorrelationsProjectedTwoPoint',
           'DEEP2StellarMassTwoPoint']


def redshift2dist(z, cosmology):
    """ Convert redshift to comoving distance in units Mpc/h.

    Parameters
    ----------
    z : float array like
    cosmology : astropy.cosmology instance

    Returns
    -------
    float array like of comoving distances
    """
    return cosmology.comoving_distance(z).to('Mpc').value * cosmology.h


class CorrelationUtilities(BaseValidationTest):
    """
    Base class for Correlation classes that loads catalogs, cuts a catalog
    sample, plots the correlation results, and scores the the results of the
    correlation measurements by comparing them to test data.

    Init of the function takes in a loaded yaml file containing the settings
    for this tests. See the following file for an example:
    descqa/configs/tpcf_Zehavi2011_rSDSS.yaml
    """
    # pylint: disable=super-init-not-called,abstract-method
    def __init__(self, **kwargs):
        self.test_name = kwargs['test_name']

        self.requested_columns = kwargs['requested_columns']
        self.test_samples = kwargs['test_samples']
        self.test_sample_labels = kwargs['test_sample_labels']

        self.output_filename_template = kwargs['output_filename_template']

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)
        self.data_label = kwargs['data_label']
        self.test_data = kwargs['test_data']

        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.fig_subplots_nrows, self.fig_subplots_ncols = kwargs.get('fig_subplots', (1, 1))
        self.fig_subplot_groups = kwargs.get('fig_subplot_groups', [None])

        self.treecorr_config = {
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)


    @staticmethod
    def load_catalog_data(catalog_instance, requested_columns, test_samples):
        """ Load requested columns from a Generic Catalog Reader instance and
        trim to the min and max of the requested cuts in test_samples.

        Parameters
        ----------
        catalog_instance : a Generic Catalog object.
        requested_columns : dictionary of lists of strings
            A dictionary keyed on a simple column name (e.g. mag, z)
            with values of lists containing string names to try to load from
            the GCR catalog instance.
            Example:
                {'mag': ['Mag_true_r_sdss_z0', 'Mag_true_r_des_z0'], ...}
        test_samples : dictionary of dictionaries
            Dictionaries containing simple column names and min max values to
            cut on.
            Examples:
                {'Mr_-23_-22": {'mag': {'min':    -23, 'max': -22}
                                'z':   {'min': 0.1031, 'max': 0.2452}}}

        Returns
        -------
        GRC catalog instance containing simplified column names and cut to the
        min/max of all requested test samples.
        """
        colnames = dict()
        for col_key, possible_names in requested_columns.items():
            colnames[col_key] = catalog_instance.first_available(*possible_names)

        if not all(v for v in colnames.values()):
            return None

        col_value_mins = defaultdict(list)
        col_value_maxs = defaultdict(list)
        for conditions in test_samples.values():
            for col_key, condition in conditions.items():
                if not isinstance(condition, dict):
                    continue
                if 'min' in condition:
                    col_value_mins[col_key].append(condition['min'])
                if 'max' in condition:
                    col_value_maxs[col_key].append(condition['max'])

        filters = [(np.isfinite, c) for c in colnames.values()]
        for col_key, col_name in colnames.items():
            if col_key in col_value_mins and col_value_mins[col_key]:
                filters.append('{} >= {}'.format(col_name, min(col_value_mins[col_key])))
            if col_key in col_value_maxs and col_value_maxs[col_key]:
                filters.append('{} < {}'.format(col_name, max(col_value_maxs[col_key])))

        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        return catalog_data


    @staticmethod
    def create_test_sample(catalog_data, test_sample):
        """ Select a subset of the catalog data an input test sample.

        This function should be overloaded in inherited classes for more
        complex cuts (e.g. color cuts).

        Parameters
        ----------
        catalog_data : a GenericCatalogReader catalog instance
        test_sample : dictionary of dictionaries
            A dictionary specifying the columns to cut on and the min/max values of
            the cut.
            Example:
                {mag: {min: -23,    max: -22}
                 z:   {min: 0.1031, max: 0.2452}}

        Returns
        -------
        A GenericCatalogReader catalog instance cut to the requested bounds.
        """
        filters = []
        for key, condition in test_sample.items():
            if isinstance(condition, dict):
                if 'max' in condition:
                    filters.append('{} < {}'.format(key, condition['max']))
                if 'min' in condition:
                    filters.append('{} >= {}'.format(key, condition['min']))
            else: #customized filter
                filters.append(condition)
        return GCRQuery(*filters).filter(catalog_data)


    def plot_data_comparison(self, corr_data, catalog_name, output_dir):
        """ Plot measured correlation functions and compare them against test
        data.

        Parameters
        ----------
        corr_data : list of float array likes
            List containing resultant data from correlation functions computed
            in the test.
            Example:
                [[np.array([...]), np.array([...]), np.array([...])], ...]
        catalog_name : string
            Name of the catalog used in the test.
        output_dir : string
            Full path of the directory to write results to.
        """
        # pylint: disable=no-member
        fig, ax_all = plt.subplots(self.fig_subplots_nrows, self.fig_subplots_ncols, squeeze=False)

        for ax, this_group in zip(ax_all.flat, self.fig_subplot_groups):
            if this_group is None:
                this_group = self.test_samples
            colors = plt.cm.plasma_r(np.linspace(0.1, 1, len(this_group)))

            for sample_name, color in zip(this_group, colors):
                try:
                    sample_corr = corr_data[sample_name]
                except KeyError:
                    continue
                sample_data = self.test_data[sample_name]
                sample_label = self.test_sample_labels.get(sample_name)

                ax.loglog(self.validation_data[:, 0],
                        self.validation_data[:, sample_data['data_col']],
                        c=color,
                        label=sample_label)
                if 'data_err_col' in sample_data:
                    y1 = (self.validation_data[:, sample_data['data_col']] +
                        self.validation_data[:, sample_data['data_err_col']])
                    y2 = (self.validation_data[:, sample_data['data_col']] -
                        self.validation_data[:, sample_data['data_err_col']])
                    y2[y2 <= 0] = self.fig_ylim[0]*0.9
                    ax.fill_between(self.validation_data[:, 0], y1, y2, lw=0, color=color, alpha=0.2)
                ax.errorbar(sample_corr[0], sample_corr[1], sample_corr[2], marker='o', ls='', c=color)

            ax.legend(loc='best')
            ax.set_xlabel(self.fig_xlabel)
            ax.set_ylim(*self.fig_ylim)
            ax.set_ylabel(self.fig_ylabel)
            ax.set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
        plt.close(fig)


    @staticmethod
    def score_and_test(corr_data): # pylint: disable=unused-argument
        """ Given the resultant correlations, compute the test score and return
        a TestResult

        Parameters
        ----------
        corr_data : list of float array likes
            List containing resultant data from correlation functions computed
            in the test.
            Example:
                [[np.array([...]), np.array([...]), np.array([...])], ...]

        Returns
        -------
        descqa.TestResult
        """
        return TestResult(inspect_only=True)


class CorrelationsAngularTwoPoint(CorrelationUtilities):
    """
    Validation test for an angular 2pt correlation function.
    """
    def __init__(self, **kwargs):
        super(CorrelationsAngularTwoPoint, self).__init__(**kwargs)
        self.treecorr_config['metric'] = 'Arc'
        self.treecorr_config['sep_units'] = 'deg'


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        catalog_data = self.load_catalog_data(catalog_instance=catalog_instance,
                                              requested_columns=self.requested_columns,
                                              test_samples=self.test_samples)

        if not catalog_data:
            return TestResult(skipped=True, summary='Missing requested quantities')

        rand_cat, rr = self.generate_processed_randoms(catalog_data)

        correlation_data = dict()
        for sample_name, sample_conditions in self.test_samples.items():
            tmp_catalog_data = self.create_test_sample(
                catalog_data, sample_conditions)

            with open(os.path.join(output_dir, 'galaxy_count.dat'), 'a') as f:
                f.write('{} {}\n'.format(sample_name, len(tmp_catalog_data['ra'])))

            if not len(tmp_catalog_data['ra']):
                continue

            output_treecorr_filepath = os.path.join(
                output_dir, self.output_filename_template.format(sample_name))

            xi_rad, xi, xi_sig = self.run_treecorr(
                catalog_data=tmp_catalog_data,
                treecorr_rand_cat=rand_cat,
                rr=rr,
                output_file_name=output_treecorr_filepath)

            correlation_data[sample_name] = (xi_rad, xi, xi_sig)

        self.plot_data_comparison(corr_data=correlation_data,
                                  catalog_name=catalog_name,
                                  output_dir=output_dir)

        return self.score_and_test(correlation_data)


    def generate_processed_randoms(self, catalog_data):
        """ Create and process random data for the 2pt correlation function.

        Parameters
        ----------
        catalog_data : dict

        Returns
        -------
        tuple of (random catalog treecorr.Catalog instance,
                  processed treecorr.NNCorrelation on the random catalog)
        """
        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size * self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside,
        )
        rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='deg', dec_units='deg')
        rr = treecorr.NNCorrelation(**self.treecorr_config)
        rr.process(rand_cat)

        return rand_cat, rr


    def run_treecorr(self, catalog_data, treecorr_rand_cat, rr, output_file_name):
        """ Run treecorr on input catalog data and randoms.

        Produce measured correlation functions using the Landy-Szalay
        estimator.

        Parameters
        ----------
        catalog_data : a GCR catalog instance
        treecorr_rand_cat : treecorr.Catalog
            Catalog of random positions over the same portion of sky as the
            input catalog_data.
        rr : treecorr.NNCorrelation
            A processed NNCorrelation of the input random catalog.
        output_file_name : string
            Full path name of the file to write the resultant correlation to.

        Returns
        -------
        tuple of array likes
           Resultant correlation function. (separation, amplitude, amp_err).
        """
        cat = treecorr.Catalog(
            ra=catalog_data['ra'],
            dec=catalog_data['dec'],
            ra_units='deg',
            dec_units='deg',
        )

        dd = treecorr.NNCorrelation(**self.treecorr_config)
        dr = treecorr.NNCorrelation(**self.treecorr_config)
        rd = treecorr.NNCorrelation(**self.treecorr_config)

        dd.process(cat)
        dr.process(treecorr_rand_cat, cat)
        rd.process(cat, treecorr_rand_cat)

        dd.write(output_file_name, rr, dr, rd)

        xi, var_xi = dd.calculateXi(rr, dr, rd)
        xi_rad = np.exp(dd.meanlogr)
        xi_sig = np.sqrt(var_xi)

        return xi_rad, xi, xi_sig


class CorrelationsProjectedTwoPoint(CorrelationUtilities):
    """
    Validation test for an radial 2pt correlation function.
    """

    def __init__(self, **kwargs):
        super(CorrelationsProjectedTwoPoint, self).__init__(**kwargs)
        self.pi_maxes = kwargs['pi_maxes']
        self.treecorr_config['metric'] = 'Rperp'


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        catalog_data = self.load_catalog_data(catalog_instance=catalog_instance,
                                              requested_columns=self.requested_columns,
                                              test_samples=self.test_samples)

        if not catalog_data:
            return TestResult(skipped=True, summary='Missing requested quantities')

        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size*self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside,
        )

        correlation_data = dict()
        for sample_name, sample_conditions in self.test_samples.items():

            output_treecorr_filepath = os.path.join(
                output_dir, self.output_filename_template.format(sample_name))

            tmp_catalog_data = self.create_test_sample(
                catalog_data, sample_conditions)

            with open(os.path.join(output_dir, 'galaxy_count.dat'), 'a') as f:
                f.write('{} {}\n'.format(sample_name, len(tmp_catalog_data['ra'])))

            if not len(tmp_catalog_data['ra']):
                continue

            xi_rad, xi, xi_sig = self.run_treecorr_projected(
                catalog_data=tmp_catalog_data,
                rand_ra=rand_ra,
                rand_dec=rand_dec,
                cosmology=catalog_instance.cosmology,
                pi_max=self.pi_maxes[sample_name],
                output_file_name=output_treecorr_filepath)

            correlation_data[sample_name] = (xi_rad, xi, xi_sig)

        self.plot_data_comparison(corr_data=correlation_data,
                                  catalog_name=catalog_name,
                                  output_dir=output_dir)

        return self.score_and_test(correlation_data)


    def run_treecorr_projected(self, catalog_data, rand_ra, rand_dec,
                               cosmology, pi_max, output_file_name):
        """ Run treecorr on input catalog data and randoms.

        Produce measured correlation functions using the Landy-Szalay
        estimator.

        Parameters
        ----------
        catalog_data : a GCR catalog instance
        rand_ra : float array like
            Random RA positions on the same sky as covered by catalog data.
        rand_dec : float array like
            Random DEC positions on the same sky as covered by catalog data.
        cosmology : astropy.cosmology
            An astropy.cosmology instance specifying the catalog cosmology.
        pi_max : float
            Maximum comoving distance along the line of sight to correlate.
        output_file_name : string
            Full path name of the file to write the resultant correlation to.

        Returns
        -------
        tuple of array likes
           Resultant correlation function. (separation, amplitude, amp_err).
        """
        treecorr_config = self.treecorr_config.copy()
        treecorr_config['min_rpar'] = -pi_max
        treecorr_config['max_rpar'] = pi_max

        cat = treecorr.Catalog(
            ra=catalog_data['ra'],
            dec=catalog_data['dec'],
            ra_units='deg',
            dec_units='deg',
            r=redshift2dist(catalog_data['z'], cosmology),
        )

        z_min = catalog_data['z'].min()
        z_max = catalog_data['z'].max()

        rand_cat = treecorr.Catalog(
            ra=rand_ra,
            dec=rand_dec,
            ra_units='deg',
            dec_units='deg',
            r=generate_uniform_random_dist(
                rand_ra.size, *redshift2dist(np.array([z_min, z_max]), cosmology)),
        )

        dd = treecorr.NNCorrelation(treecorr_config)
        dr = treecorr.NNCorrelation(treecorr_config)
        rd = treecorr.NNCorrelation(treecorr_config)
        rr = treecorr.NNCorrelation(treecorr_config)

        dd.process(cat)
        dr.process(rand_cat, cat)
        rd.process(cat, rand_cat)
        rr.process(rand_cat)

        dd.write(output_file_name, rr, dr, rd)

        xi, var_xi = dd.calculateXi(rr, dr, rd)
        xi_rad = np.exp(dd.meanlogr)
        xi_sig = np.sqrt(var_xi)

        return xi_rad, xi * 2. * pi_max, xi_sig * 2. * pi_max


class DEEP2StellarMassTwoPoint(CorrelationsProjectedTwoPoint):
    """ Test simulated data against the power laws fits to Stellar Mass
    selected samples in DEEP2. This class also serves as an example of creating
    a specific test from the two correlation classes in the test suite.

    In the future this could also include a color cut, however absolute U and B
    band magnitudes are not stored in the simulated catalogs currently and
    converting the current fluxes to those is currently out of scope.
    """
    @staticmethod
    def power_law(r, r0, g):
        """ Compute the power law of a simple 2 parameter projected correlation
        function.

        Parameters
        ---------
        r : float array like
            Comoving positions to compute the power law at.
        r0 : float
            Amplitude of the correlation function
        g : float
            Power law of the correlation function.

        Returns
        -------
        float array like
        """
        gamma_func_ratio = scsp.gamma(1/2.) * scsp.gamma((g - 1) / 2) / scsp.gamma(g / 2)
        return r * (r0 / r) ** g * gamma_func_ratio


    @staticmethod
    def power_law_err(r, r0, g, r0_err, g_err):
        """ Compute the error on the power law model given errors on r0 and g.
        function.

        Parameters
        ---------
        r : float array like
            Comoving positions to compute the power law at.
        r0 : float
            Amplitude of the correlation function
        g : float
            Power law of the correlation function.
        r0_err : float
            Error on r0
        g_err : float
            Error on the power law slope.

        Returns
        -------
        float array like
        """
        gamma_func_ratio = scsp.gamma(1/2.) * scsp.gamma((g - 1) / 2) / scsp.gamma(g / 2)
        p_law = r * (r0 / r) ** g * gamma_func_ratio
        dev_r0 = r ** (1 - g) * r0 ** (g - 1) * g * gamma_func_ratio * r0_err
        dev_g = (p_law * np.log(r) +
                 2 * p_law * scsp.polygamma(0, (g - 1) / 2) +
                 -2 * p_law * scsp.polygamma(0, g / 2)) * g_err
        return np.sqrt(dev_r0 ** 2 + dev_g ** 2)


    def plot_data_comparison(self, corr_data, catalog_name, output_dir):
        fig, ax = plt.subplots()
        colors = plt.cm.plasma_r(np.linspace(0.1, 1, len(self.test_samples))) # pylint: disable=no-member

        for sample_name, color in zip(self.test_samples, colors):
            sample_corr = corr_data[sample_name]
            sample_data = self.test_data[sample_name]
            sample_label = self.test_sample_labels.get(sample_name)

            p_law = self.power_law(sample_corr[0],
                                   self.validation_data[sample_data['row'],
                                                        sample_data['r0']],
                                   self.validation_data[sample_data['row'],
                                                        sample_data['g']])
            p_law_err = self.power_law_err(sample_corr[0],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['r0']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['g']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['r0_err']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['g_err']])
            ax.loglog(sample_corr[0],
                      p_law,
                      c=color,
                      label=sample_label)
            ax.fill_between(sample_corr[0],
                            p_law - p_law_err,
                            p_law + p_law_err,
                            lw=0, color=color, alpha=0.2)
            ax.errorbar(sample_corr[0], sample_corr[1], sample_corr[2], marker='o', ls='', c=color)

        ax.legend(loc='best')
        ax.set_xlabel(self.fig_xlabel)
        ax.set_ylim(*self.fig_ylim)
        ax.set_ylabel(self.fig_ylabel)
        ax.set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
        plt.close(fig)


    def score_and_test(self, corr_data):
        """ Test the average chi^2 per degree of freedom against power law fits
        to the DEEP2 dataset.
        """
        chi_per_nu = 0
        total_sample = 0
        rbins = list(corr_data.values()).pop()[0]
        r_idx_min = np.searchsorted(rbins, 1.)
        r_idx_max = np.searchsorted(rbins, 10., side='right')
        for sample_name in self.test_samples:
            sample_corr = corr_data[sample_name]
            sample_data = self.test_data[sample_name]

            r_data = sample_corr[0][r_idx_min:r_idx_max]
            p_law = self.power_law(r_data,
                                   self.validation_data[sample_data['row'],
                                                        sample_data['r0']],
                                   self.validation_data[sample_data['row'],
                                                        sample_data['g']])
            p_law_err = self.power_law_err(r_data,
                                           self.validation_data[sample_data['row'],
                                                                sample_data['r0']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['g']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['r0_err']],
                                           self.validation_data[sample_data['row'],
                                                                sample_data['g_err']])
            chi_per_nu = np.sum(((sample_corr[1][r_idx_min:r_idx_max] - p_law) / p_law_err) ** 2)
            chi_per_nu /= len(r_data)
            total_sample += 1

        score = chi_per_nu / total_sample
        # Made up value. Assert that average chi^2/nu is less than 2.
        test_pass = score < 2

        return TestResult(score=score,
                          passed=test_pass,
                          summary="Ave chi^2/nu value comparing to power law fits to stellar mass threshold "
                                  "DEEP2 data. Test threshold set to 2.")
