from __future__ import print_function, division, unicode_literals, absolute_import
import os
import numpy as np
from GCR import GCRQuery
import treecorr

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .utils import generate_uniform_random_ra_dec_footprint, get_healpixel_footprint, generate_uniform_random_dist


__all__ = ['CorrelationsTwoPoint']


def redshift2dist(z, cosmology):
    return cosmology.comoving_distance(z).to('Mpc').value * cosmology.h


class CorrelationsTwoPoint(BaseValidationTest):
    """
    Validation test of 2pt correlation function
    """
    _C = 299792.458

    def __init__(self, **kwargs):
        self.possible_mag_fields = kwargs['possible_mag_fields']
        self.need_distance = kwargs['need_distance']
        self.data_label = kwargs['data_label']
        self.mag_bins = kwargs['mag_bins']
        self.output_filename_template = kwargs['output_filename_template']
        self.label_template = kwargs['label_template']
        self.fig_xlabel = kwargs['fig_xlabel']
        self.fig_ylabel = kwargs['fig_ylabel']
        self.test_name = kwargs['test_name']

        self.random_nside = kwargs.get('random_nside', 1024)
        self.random_mult = kwargs.get('random_mult', 3)

        self._treecorr_config = {
            'metric': ('Rperp' if self.need_distance else 'Arc'),
            'min_sep': kwargs['min_sep'],
            'max_sep': kwargs['max_sep'],
            'bin_size': kwargs['bin_size'],
        }
        if not self.need_distance:
            self._treecorr_config['sep_units'] = 'deg'

        validation_filepath = os.path.join(self.data_dir, kwargs['data_filename'])
        self.validation_data = np.loadtxt(validation_filepath, skiprows=2)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        '''
        Loop over magnitude cuts and make plots
        '''
        # load catalog data
        colnames = dict()
        colnames['z'] = catalog_instance.first_available('redshift', 'redshift_true')
        colnames['ra'] = catalog_instance.first_available('ra', 'ra_true')
        colnames['dec'] = catalog_instance.first_available('dec', 'dec_true')
        colnames['mag'] = catalog_instance.first_available(*self.possible_mag_fields)

        if not all(v for v in colnames.values()):
            return TestResult(skipped=True, summary='Missing requested quantities')

        filters = [(np.isfinite, c) for c in colnames.values()]
        filters.extend((
            '{} < {}'.format(colnames['mag'], max(mag_bin['mag_max'] for mag_bin in self.mag_bins)),
            '{} >= {}'.format(colnames['mag'], min(mag_bin['mag_min'] for mag_bin in self.mag_bins)),
        ))
        if self.need_distance:
            filters.extend((
                '{} < {}'.format(colnames['z'], max(mag_bin['cz_max'] for mag_bin in self.mag_bins)/self._C),
                '{} >= {}'.format(colnames['z'], min(mag_bin['cz_min'] for mag_bin in self.mag_bins)/self._C),
            ))
        catalog_data = catalog_instance.get_quantities(list(colnames.values()), filters=filters)
        catalog_data = {k: catalog_data[v] for k, v in colnames.items()}

        # create random
        rand_ra, rand_dec = generate_uniform_random_ra_dec_footprint(
            catalog_data['ra'].size*self.random_mult,
            get_healpixel_footprint(catalog_data['ra'], catalog_data['dec'], self.random_nside),
            self.random_nside,
        )

        if not self.need_distance:
            rand_cat = treecorr.Catalog(ra=rand_ra, dec=rand_dec, ra_units='deg', dec_units='deg')
            del rand_ra, rand_dec
            rr = treecorr.NNCorrelation(**self._treecorr_config)
            rr.process(rand_cat)


        fig, ax = plt.subplots()
        try:
            for mag_bin, color in zip(self.mag_bins, plt.cm.plasma_r(np.linspace(0.1, 1, len(self.mag_bins)))):

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
                    r=(redshift2dist(catalog_data_this['z'], catalog_instance.cosmology) if self.need_distance else None),
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
                            *redshift2dist(np.array([mag_bin['cz_min'], mag_bin['cz_max']])/self._C, catalog_instance.cosmology)
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

                output_filepath = os.path.join(output_dir, self.output_filename_template.format(mag_bin['mag_min'], mag_bin['mag_max']))
                dd.write(output_filepath, rr, dr, rd)

                xi, var_xi = dd.calculateXi(rr, dr, rd)
                xi_rad = np.exp(dd.meanlogr)
                xi_sig = np.sqrt(var_xi)


                ax.loglog(self.validation_data[:,0], self.validation_data[:,mag_bin['data_col']], c=color, label=self.label_template.format(mag_bin['mag_min'], mag_bin['mag_max']))
                if 'data_err_col' in mag_bin:
                    ax.fill_between(
                        self.validation_data[:,0],
                        self.validation_data[:,mag_bin['data_col']]+self.validation_data[:,mag_bin['data_err_col']],
                        self.validation_data[:,mag_bin['data_col']]-self.validation_data[:,mag_bin['data_err_col']],
                        color=color,
                        alpha=0.2)
                scale_wp = mag_bin['pi_max'] * 2.0 if 'pi_max' in mag_bin else 1.0
                ax.errorbar(xi_rad, xi*scale_wp, xi_sig*scale_wp, marker='o', ls='', c=color)

            ax.legend(loc='best')
            ax.set_xlabel(self.fig_xlabel)
            ax.set_ylabel(self.fig_ylabel)
            ax.set_title('{} vs. {}'.format(catalog_name, self.data_label), fontsize='medium')

        finally:
            fig.savefig(os.path.join(output_dir, '{:s}.png'.format(self.test_name)), bbox_inches='tight')
            plt.close(fig)

        #TODO: calculate summary statistics
        return TestResult(0, passed=True)
