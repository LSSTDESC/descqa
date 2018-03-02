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


__all__ = ['CorrelationsTwoPoint']


def redshift2dist(z, cosmology):
    return cosmology.comoving_distance(z).to('Mpc').value * cosmology.h


class CorrelationsTwoPoint(BaseValidationTest):
    """
    Validation test of 2pt correlation function
    """
    _C = 299792.458

    def __init__(self, **kwargs):
        self.requsted_columns = kwargs['requsted_columns']
        self.test_samples = kwargs['test_stamples']

        self.need_distance = kwargs['need_distance']
        self.data_label = kwargs['data_label']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.fig_ylim = kwargs['fig_ylim']
        self.test_name = kwargs['test_name']

        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)

        self._treecorr_config = {
            'metric': ('Rperp' if self.need_distance else 'Arc'),
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        self.pi_max = kwargs.get('pi_max', None)
        if not self.need_distance:
            self._treecorr_config['sep_units'] = 'deg'

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)

    def load_catalog_data(self, catalog_instance):
        """
        """

        colnames = dict()
        col_value_mins = dict()
        col_value_maxs = dict()
        for key in self.requsted_columns.keys():
            colnames[key] = catalog_instance.first_available(*self.requsted_columns[key])
            col_value_mins[key] = []
            col_value_maxs[key] = []
            for sample in test_samples.keys():
                col_value_mins[key].append(test_samples[sample][key]['min'])
                col_value_maxs[key].append(test_samples[sample][key]['max'])

        if not all(v for v in colnames.values()):
            return TestResult(skipped=True, summary='Missing requested quantities')

        filters = [(np.isfinite, c) for c in colnames.values()]

        for key in self.requsted_columns.keys():
            filters.extend((
                '{} < {}'.format(colnames[key], max(col_value_maxs[key])),
                '{} >= {}'.format(colnames[key], min(col_value_mins[key])),
            ))

        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        return catalog_data

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''

        catalog_data = load_catalog_data(catalog_instance)

        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size * self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside,
        )
        if not self.need_distance:
            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='deg', dec_units='deg')
            del rand_ra, rand_dec
            rr = treecorr.NNCorrelation(**self._treecorr_config)
            rr.process(rand_cat)

        for sample in test_samples:
            filters = []
            for key in sample.keys():
                filters.extend([
                    '{} < {}'.format(key, sample[key]),
                    '{} >= {}'.format(key, sample[key]),
                ])
            catalog_data_this = GCRQuery(*filters).filter(catalog_data)

            cat = treecorr.Catalog(
                ra=catalog_data_this['ra'],
                dec=catalog_data_this['dec'],
                ra_units='deg',
                dec_units='deg',
                r=(redshift2dist(catalog_data_this['z'], catalog_instance.cosmology)
                    if self.need_distance else None),
            )

            del catalog_data_this

            xi_rad, xi_sig = run_tree_corr(cat, )

        fig, ax = plt.subplots()
        try:
            for mag_bin, color in zip(self.mag_bins,
                                      plt.cm.plasma_r(np.linspace(0.1, 1, len(self.mag_bins)))):

                # filter catalog data for this bin
                filters = [
                    'mag < {}'.format(mag_bin['mag_max']),
                    'mag >= {}'.format(mag_bin['mag_min']),
                ]
                if self.need_distance:
                    filters.extend((
                        'z < {}'.format(mag_bin['cz_max']/self._C),
                        'z >= {}'.format(mag_bin['cz_min']/self._C),
                    ))

                catalog_data_this = GCRQuery(*filters).filter(catalog_data)

                cat = treecorr.Catalog(
                    ra=catalog_data_this['ra'],
                    dec=catalog_data_this['dec'],
                    ra_units='deg',
                    dec_units='deg',
                    r=(redshift2dist(catalog_data_this['z'], catalog_instance.cosmology)
                       if self.need_distance else None),
                )

                del catalog_data_this

                treecorr_config = self._treecorr_config.copy()
                if 'pi_max' in mag_bin:
                    treecorr_config['min_rpar'] = -mag_bin['pi_max']
                    treecorr_config['max_rpar'] = mag_bin['pi_max']

                if self.need_distance:
                    rand_cat = treecorr.Catalog(
                        ra=rand_ra,
                        dec=rand_dec,
                        ra_units='deg',
                        dec_units='deg',
                        r=generate_uniform_random_dist(
                            rand_ra.size,
                            *redshift2dist(np.array([mag_bin['cz_min'], mag_bin['cz_max']]) /
                                           self._C, catalog_instance.cosmology)
                        ),
                    )
                    rr = treecorr.NNCorrelation(treecorr_config)
                    rr.process(rand_cat)

                dd = treecorr.NNCorrelation(treecorr_config)
                dr = treecorr.NNCorrelation(treecorr_config)
                rd = treecorr.NNCorrelation(treecorr_config)

                dd.process(cat)
                dr.process(rand_cat, cat)
                rd.process(cat, rand_cat)

                output_filepath = os.path.join(
                    output_dir, self.output_filename_template.format(mag_bin['mag_min'], mag_bin['mag_max']))
                dd.write(output_filepath, rr, dr, rd)

                xi, var_xi = dd.calculateXi(rr, dr, rd)
                xi_rad = np.exp(dd.meanlogr)
                xi_sig = np.sqrt(var_xi)


                ax.loglog(self.validation_data[:,0],
                          self.validation_data[:,mag_bin['data_col']],
                          c=color,
                          label=self.label_template.format(mag_bin['mag_min'], mag_bin['mag_max']))
                if 'data_err_col' in mag_bin:
                    y1 = (self.validation_data[:,mag_bin['data_col']] +
                          self.validation_data[:,mag_bin['data_err_col']])
                    y2 = (self.validation_data[:,mag_bin['data_col']] -
                          self.validation_data[:,mag_bin['data_err_col']])
                    y2[y2<=0] = self.fig_ylim[0]*0.9
                    ax.fill_between(self.validation_data[:,0], y1, y2, lw=0, color=color, alpha=0.2)
                scale_wp = mag_bin['pi_max'] * 2.0 if 'pi_max' in mag_bin else 1.0
                ax.errorbar(xi_rad, xi*scale_wp, xi_sig*scale_wp, marker='o', ls='', c=color)

            ax.legend(loc='best')
            ax.set_xlabel(self.fig_xlabel)
            ax.set_ylim(*self.fig_ylim)
            ax.set_ylabel(self.fig_ylabel)
            ax.set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')

        finally:
            fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
            plt.close(fig)

        #TODO: calculate summary statistics
        return TestResult(inspect_only=True)
