from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
import treecorr

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import generate_uniform_random_ra_dec_footprint, \
                   get_healpixel_footprint, \
                   generate_uniform_random_dist

__all__ = ['CorrelationsAngularTwoPoint', 'CorrelationsProjectedTwoPoint']


def redshift2dist(z, cosmology):
    """ Convert redshift to comoving distance in units Mpc/h.

    Parameters
    ----------
    z : float array like
    cosmology : astropy.cosmology instance

    Returns
    -------
    float array like comoving distances
    """
    return cosmology.comoving_distance(z).to('Mpc').value * cosmology.h


def load_catalog_data(catalog_instance, requested_columns, test_samples):
        """ Load requested columns from a Generic Catalog Reader instance.

        Parameters
        ----------
        catalog_instance : a Generic Catalog object.
        rested_columns : dictionary of lists of strings
            A dictionary containing keyed on a simple column name (e.g. mag, z)
            with values of lists containing string names to try to load from
            the input GCR.
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
        col_value_mins = dict()
        col_value_maxs = dict()
        for key in requested_columns.keys():
            colnames[key] = catalog_instance.first_available(*requested_columns[key])
            if key != 'ra' and key != 'dec':
                col_value_mins[key] = []
                col_value_maxs[key] = []
                for sample in test_samples.keys():
                    col_value_mins[key].append(test_samples[sample][key]['min'])
                    col_value_maxs[key].append(test_samples[sample][key]['max'])

        if not all(v for v in colnames.values()):
            return TestResult(skipped=True, summary='Missing requested quantities')

        filters = [(np.isfinite, c) for c in colnames.values()]

        for key in requested_columns.keys():
            if key != 'ra' and key != 'dec':
                filters.extend((
                    '{} < {}'.format(colnames[key], max(col_value_maxs[key])),
                    '{} >= {}'.format(colnames[key], min(col_value_mins[key])),
                ))

        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        return catalog_data


def create_test_sample(catalog_data, test_sample):
    """ Select a subset of the catalog data an input test sample.

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
    for key in test_sample.keys():
        filters.extend((
            '{} < {}'.format(key, test_sample[key]['max']),
            '{} >= {}'.format(key, test_sample[key]['min']),
        ))
    return GCRQuery(*filters).filter(catalog_data)


class PlotCorrelation(object):
    """ Mixin class for plotting the results of the correlation measurements
    and comparing them to test data.
    """

    def plot_data_comparison(self, corr_data, catalog_name, output_dir):
        """ Plot measured correlation functions and compare them against test
        data.

        Parameters
        ----------
        corr_data : list of float array likes
            List containing resultant data from correlation functions computed
            in the test.
            Example:
                [[np.array([...]), np.array([...]), np.array([...])]]
        catalog_name : string
            Name of the catalog used in the test.
        output_dir : string
            Full path of the directory to write results to.
        """
        fig, ax = plt.subplots()

        for sample_name, sample_corr, color in zip(self.test_samples.keys(),
                                                   corr_data,
                                                   plt.cm.plasma_r(
                                                       np.linspace(0.1, 1, len(self.test_samples)))):

            ax.loglog(self.validation_data[:, 0],
                      self.validation_data[:, self.test_data[sample_name]['data_col']],
                      c=color,
                      label=self.label_template.format(
                          self.test_samples[sample_name][self.label_column]['min'],
                          self.test_samples[sample_name][self.label_column]['max']))
            if 'data_err_col' in self.test_data[sample_name]:
                y1 = (self.validation_data[:, self.test_data[sample_name]['data_col']] +
                      self.validation_data[:, self.test_data[sample_name]['data_err_col']])
                y2 = (self.validation_data[:, self.test_data[sample_name]['data_col']] -
                      self.validation_data[:, self.test_data[sample_name]['data_err_col']])
                y2[y2 <= 0] = self.fig_ylim[0]*0.9
                ax.fill_between(self.validation_data[:, 0], y1, y2, lw=0, color=color, alpha=0.2)
            ax.errorbar(sample_corr[0], sample_corr[1], sample_corr[2], marker='o', ls='', c=color)

        ax.legend(loc='best')
        ax.set_xlabel(self.fig_xlabel)
        ax.set_ylim(*self.fig_ylim)
        ax.set_ylabel(self.fig_ylabel)
        ax.set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')

        fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
        plt.close(fig)


class CorrelationsAngularTwoPoint(BaseValidationTest, PlotCorrelation):
    """
    Validation test for an angular 2pt correlation function.

    Init of the function takes in a loaded yaml file containing the settings
    for this tests. See the following file for an example:
    descqa/configs/tpcf_Wang2013_rSDSS.yaml
    """
    def __init__(self, **kwargs): #pylint: disable=W0231
        self.test_name = kwargs['test_name']

        self.requested_columns = kwargs['requested_columns']
        self.test_samples = kwargs['test_samples']

        self.output_filename_template = kwargs['output_filename_template']

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)
        self.data_label = kwargs['data_label']
        self.test_data = kwargs['test_data']

        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.label_column = kwargs['label_column']
        self.label_template = kwargs['label_template']

        self.treecorr_config = {
            'metric': 'Arc',
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        self.treecorr_config['sep_units'] = 'deg'
        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        catalog_data = load_catalog_data(catalog_instance=catalog_instance,
                                         requested_columns=self.requested_columns,
                                         test_samples=self.test_samples)

        rand_cat, rr = self.generate_processed_randoms(catalog_data)

        correlation_data = []
        for sample_name in self.test_samples.keys():
            tmp_catalog_data = self.create_test_sample(catalog_data, test_samples[sample_name])

            output_treecorr_filepath = os.path.join(
                output_dir, self.output_filename_template.format(sample_name))

            xi_rad, xi, xi_sig = self.run_treecorr(
                catalog_data=tmp_catalog_data,
                treecorr_rand_cat=rand_cat,
                rr=rr,
                output_filepath=output_treecorr_filepath)
            correlation_data.append([xi_rad, xi, xi_sig])

        self.plot_data_comparison(corr_data=correlation_data,
                                  catalog_name=catalog_name,
                                  output_filename=output_dir)

        return TestResult(inspect_only=True)

    def generate_processed_randoms(self, catalog_data):
        """ Create and process random data for the 2pt correlation function.

        Parameters
        ----------
        catalog_data : a GRC catalog instance

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

    def run_tree_corr(self, catalog_data, treecorr_rand_cat, rr, output_file_name):
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
        list of array likes
           Resultant correlation function. separation, amplitude, amp_err.
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

    def compute_summary_statistics(self):
        """
        """
        pass


class CorrelationsProjectedTwoPoint(BaseValidationTest, PlotCorrelation):
    """
    Validation test for an radial 2pt correlation function.

    Init of the function takes in a loaded yaml file containing the settings
    for this tests. See the following file for an example:
    descqa/configs/tpcf_Zehavi2011_rSDSS.yaml
    """

    def __init__(self, **kwargs): #pylint: disable=W0231
        self.test_name = kwargs['test_name']

        self.requested_columns = kwargs['requested_columns']
        self.test_samples = kwargs['test_samples']

        self.output_filename_template = kwargs['output_filename_template']

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)
        self.data_label = kwargs['data_label']
        self.test_data = kwargs['test_data']

        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.label_column = kwargs['label_column']
        self.label_template = kwargs['label_template']

        self.pi_maxes = kwargs['pi_maxes']

        self.treecorr_config = {
            'metric': 'Rperp',
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        catalog_data = load_catalog_data(catalog_instance=catalog_instance,
                                         requested_columns=self.requested_columns,
                                         test_samples=self.test_samples)

        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size*self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside,
        )

        correlation_data = []
        for sample_name in self.test_samples.keys():

            output_treecorr_filepath = os.path.join(
                output_dir, self.output_filename_template.format(sample_name))

            tmp_catalog_data = create_test_sample(catalog_data, self.test_samples[sample_name])

            xi_rad, xi, xi_sig = self.run_treecorr_projected(
                catalog_data=tmp_catalog_data,
                rand_ra=rand_ra,
                rand_dec=rand_dec,
                cosmology=catalog_instance.cosmology,
                z_min=self.test_samples[sample_name]['z']['min'],
                z_max=self.test_samples[sample_name]['z']['max'],
                pi_max=self.pi_maxes[sample_name],
                output_file_name=output_treecorr_filepath)
            correlation_data.append([xi_rad, xi, xi_sig])

        self.plot_data_comparison(corr_data=correlation_data,
                                  catalog_name=catalog_name,
                                  output_dir=output_dir)

        return TestResult(inspect_only=True)

    def run_treecorr_projected(self, catalog_data, rand_ra, rand_dec,
                               cosmology, z_min, z_max, pi_max, output_file_name):
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
        z_min : float
            Minimum redshift of the catalog_data sample
        z_max : float
            Maximum redshift of the catalog_data sample
        pi_max : float
            Maximum comoving distance along the line of sight to correlate
        output_file_name : string
            Full path name of the file to write the resultant correlation to.

        Returns
        -------
        list of array likes
           Resultant correlation function. separation, amplitude, amp_err.
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

    def compute_summary_statistics(self):
        """
        """
        pass
