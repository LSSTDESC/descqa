from __future__ import unicode_literals, absolute_import, division
import os
import re
import numpy as np
from scipy.stats import binned_statistic as bs
import matplotlib.colors as clr
from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['ColorRedshiftTest']
class _CatalogDoesNotHaveQuantity(Exception):
    """Raised when the catalog doesn't have a quantity and indicates the
    test should be skipped"""
    def __init__(self, quantity_name):
        super(_CatalogDoesNotHaveQuantity, self).__init__()
        self.message = "Catalog does not have {}".format(quantity_name)

class ColorRedshiftTest(BaseValidationTest):
    """
    This test plots various color-redshfit diagnostics
    """

    possible_observations = {
        'des_fit': {'filename_template':'red_sequence/des/rykoff_et_al_1026',
                'keys': (0),
                'coefficients': (1, 2, 3, 4),
                'skip': 7,
                'label': 'DES fit',
                'zmin': 0.2,
                'zmax': 0.9,
                'format': 'fit',
                },
         'des_y1':{'filename_template':'red_sequence/des/des_y1_redshift_ri_color.txt',
                   'skip': 1,
                   'usecols': (0,1),
                   'colnames': ('z', 'r-i'),
                   'label':'DES Y1',
                   'format': 'data',
                   'bins': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9],
                },
         }

    
    def __init__(self, **kwargs):
        super(ColorRedshiftTest, self).__init__()
        # load test config options
        self.kwargs = kwargs
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.title_in_legend = kwargs.get('title_in_legend', True)
        self.font_size = kwargs.get('font_size', 16)
        self.text_size = kwargs.get('text_size', 12)
        self.legend_size = kwargs.get('legend_size', 13)
        self.observations = kwargs.get('observations', None)
        
        #with open(os.path.join(self.data_dir, 'README.md')) as f:
        #    self.validation_data = f.readline().strip()
        self.plot_list = kwargs.get("plot_list", [])
        for plot_param in self.plot_list:
            color = plot_param['color']
            assert (len(color) == 3) and (color[1] == '-'), "Color must be defined as 'a-b', where a and b are band names"
            allowed_colors = 'ugrizy'
            plot_param['mag1'] = color[0].lower()
            plot_param['mag2'] = color[2].lower()
            assert (plot_param["mag1"] in allowed_colors) and (plot_param["mag2"] in allowed_colors), "only ugrizy colors are allowed"
            assert (plot_param["frame"] in ['rest', 'obs', 'observed', 'observer']), "Only 'rest', 'obs', and 'observed' frames allowed"
            plot_param["filter"] = plot_param.get('filter', '').lower()
            assert (plot_param["filter"] in ['lsst', 'sdss', 'des']), "Only lsst, sdss, or DES filters allowed"
            plot_param["baseDC2"] = plot_param.get('baseDC2', False)
            plot_param["central"] = plot_param.get("central", None)
            plot_param["Mr_cut"] = plot_param.get("Mr_cut", None)
            plot_param["mr_cut"] = plot_param.get("mr_cut", None)
            plot_param["stellar_mass_cut"] = plot_param.get("stellar_mass_cut", None)
            plot_param["halo_mass_cut"]  = plot_param.get("halo_mass_cut", None)
            plot_param["red_sequence_cut"] = plot_param.get("red_sequence_cut", None)
            plot_param["synthetic_type"] = plot_param.get("synthetic_type", None)
            plot_param["log_scale"] = plot_param.get("log_scale", True)
            plot_param["redshift_limit"] = plot_param.get("redshift_limit", None)
            plot_param["redshift_block_limit"] = plot_param.get("redshift_block_limit", 1)
            assert plot_param['redshift_block_limit'] in [1, 2, 3], "redshift_block_limit must be set to 1,2 or 3. It is set to: {}".format(plot_param['redshift_block_limit'])

        #read in validation data
        self.validation_data = self.get_validation_data(self.observations)
            
    def get_validation_data(self, observations):
        validation_data = {}
        if observations:
            for obs in observations:
                data_args = self.possible_observations[obs]
                fn = os.path.join(self.data_dir, data_args['filename_template'])
                if 'keys' in data_args.keys():
                    keys = np.genfromtxt(fn, skip_header=data_args['skip'], usecols=data_args['keys'], dtype=str)
                    coefficients = np.genfromtxt(fn, skip_header=data_args['skip'], usecols=data_args['coefficients'])
                    validation_data[obs] = dict(zip(keys, coefficients))
                else:
                    validation_data[obs] = dict(zip(data_args['colnames'],
                                               np.loadtxt(fn, skiprows=data_args['skip'],
                                                          unpack=True, usecols=data_args['usecols'])))
                validation_data[obs]['label'] = data_args['label']
                validation_data[obs]['format'] = data_args['format']
                if 'zmin' in data_args.keys():
                    validation_data[obs]['zmin'] = data_args['zmin']
                if 'zmax' in data_args.keys():
                    validation_data[obs]['zmax'] = data_args['zmax']
                if 'bins' in data_args.keys():
                    validation_data[obs]['bins'] = data_args['bins']

        return validation_data

    def post_process_plot(self, ax):
        pass
        # ax.text(0.05, 0.95, self.validation_data)
        # ax.legend(loc='best')

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        plot_num = 0
        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]
            
        for plot_param in self.plot_list:
            plot_num += 1
            if plot_param["frame"] == "rest":
                mag_frame = "Mag_true"
                mag_end = "_z0"
            else:
                mag_frame = "mag"
                mag_end = ""
            mag1_str = "{}_{}_{}{}".format(mag_frame, plot_param["mag1"],
                                           plot_param["filter"], mag_end)
            mag2_str = "{}_{}_{}{}".format(mag_frame, plot_param["mag2"],
                                           plot_param["filter"], mag_end)
            mag1_val = self._get_quantity(catalog_instance, mag1_str,
                                          redshift_block_limit=plot_param['redshift_block_limit'],
                                          redshift_limit=plot_param['redshift_limit'],)
            mag2_val = self._get_quantity(catalog_instance, mag2_str,
                                          redshift_block_limit=plot_param['redshift_block_limit'],
                                          redshift_limit=plot_param['redshift_limit'],)
            redshift = self._get_quantity(catalog_instance, 'redshift',
                                          redshift_block_limit=plot_param['redshift_block_limit'],
                                          redshift_limit=plot_param['redshift_limit'],)
            clr_val = mag1_val - mag2_val
            title = ""
            slct, title = self._get_selection_and_title(catalog_instance, title, plot_param,
                                                        redshift_limit=plot_param['redshift_limit'],
                                                        redshift_block_limit=plot_param['redshift_block_limit'])

            fig, ax = plt.subplots()
            # for ax_this in (ax, self.summary_ax):
            if plot_param['redshift_limit'] is not None:
                redshift_bins = np.linspace(0, 1.05*plot_param['redshift_limit'], 256)
            elif plot_param['redshift_block_limit'] is not None:
                redshift_bins = np.linspace(0, 1.05*(plot_param['redshift_block_limit']), 256)
            else:
                redshift_bins = np.linspace(0, 1.05, 256)

            h, xbins, ybins = np.histogram2d(redshift[slct], clr_val[slct],
                                             bins=(redshift_bins, np.linspace(-0.4, 2.2, 256)))
            if plot_param["log_scale"]:
                pc = ax.pcolor(xbins, ybins, h.T+1.0, norm=clr.LogNorm())
                fig.colorbar(pc, ax=ax).set_label("Population Density + 1")
            else:
                pc = ax.pcolor(xbins, ybins, h.T)
                fig.colorbar(pc, ax=ax).set_label("Population Density")
            mag1 = re.split('_', mag1_str)[1]  #get filter
            mag2 = re.split('_', mag2_str)[1]  #get filter
            # plot observations
            for v in self.validation_data.values():
                color=mag1 + '-' + mag2
                if v['format'] == 'fit':
                    coeffs = v[color]
                    zmask = (redshift_bins >= v['zmin']) & (redshift_bins <= v['zmax'])
                    obs = np.zeros(len(redshift_bins[zmask]))
                    for n, coeff in enumerate(coeffs):
                        obs += coeff*redshift_bins[zmask]**(len(coeffs)-1-n)

                    ax.plot(redshift_bins[zmask], obs, color='r', label=v['label'])
                elif v['format'] == 'data':
                    if color in v.keys():
                        zbins = np.asarray(v['bins'])
                        mean, edge, num = bs(v['z'], v[color], bins=zbins)
                        std, edge, num = bs(v['z'], v[color], bins=zbins, statistic='std')
                        fmask = np.isfinite(mean)
                        z_cen = 0.5*(zbins[1:]+zbins[:-1])
                        ax.errorbar(z_cen[fmask], mean[fmask], ls='', marker='o',
                                    yerr=np.sqrt(std[fmask]), c='orange', label=v['label'])
                        counts = [np.sum(num==i+1) for i in range(len(zbins))]
                        print(z_cen[fmask], mean[fmask], std[fmask], counts)
                        
                legend = ax.legend(loc='lower right', fontsize=self.legend_size)
                plt.setp(legend.get_texts(), color='w')
                
            ax.set_ylabel('{} - {}'.format(mag1,  mag2), size=self.font_size)
            ax.set_xlabel('Redshift $z$', size=self.font_size)
            if self.title_in_legend:
                title = '{}\n{}'.format(catalog_name, title)
            else:
                ax.set_title(catalog_name)
            ax.text(0.05, 0.95, title, transform=ax.transAxes,
                    verticalalignment='top', color='white',
                    fontsize=self.text_size)
            fig.savefig(os.path.join(output_dir, 'plot_{}.png'.format(plot_num)))
            plt.close(fig)


        return TestResult(0, inspect_only=True)


    def _get_quantity(self, catalog_instance, quantity_name,
                      redshift_block_limit=1,
                      redshift_limit=None):
        if not catalog_instance.has_quantities([quantity_name]):
            raise _CatalogDoesNotHaveQuantity(quantity_name)
        first_name = catalog_instance.first_available(quantity_name)
        if redshift_limit is not None:
            filters = ["redshift < {}".format(redshift_limit)]
            if redshift_limit <= 1:
                redshift_block_limit = 1
            elif redshift_limit <= 2:
                redshift_block_limit = 2
            else:
                redshift_block_limit = 3
        else:
            filters = None
        native_filters = ['redshift_block_lower <= {}'.format(redshift_block_limit-1)]
        return catalog_instance.get_quantities([first_name],
                                               filters=filters,
                                               native_filters=native_filters,)[first_name]


    def _get_selection_and_title(self, catalog_instance, title, plot_param,
                                 redshift_block_limit=1,
                                 redshift_limit=None):
        # a cheap way to get an array of trues of the correct size
        redshift = self._get_quantity(catalog_instance, 'redshift',
                                      redshift_limit=redshift_limit,
                                      redshift_block_limit=redshift_block_limit)
        slct = redshift == redshift
        title_elem_per_line = 3 # The number of elements in the title. We want about
        title_elem = 0 # three elements per line. The catalog name is pretty big, so it counts
        # as two elements.
        if plot_param["central"] is not None:
            is_central = self._get_quantity(catalog_instance, 'is_central',
                                            redshift_limit=redshift_limit,
                                            redshift_block_limit=redshift_block_limit)
            slct = slct & (is_central == plot_param["central"])
            title += "central = {}, ".format(plot_param["central"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if plot_param["Mr_cut"] is not None:
            Mag_r = self._get_quantity(catalog_instance, "Mag_true_r_lsst_z0",
                                       redshift_limit=redshift_limit,
                                       redshift_block_limit=redshift_block_limit)
            slct = slct & (Mag_r < plot_param["Mr_cut"])
            title += "Mr < {}, ".format(plot_param["Mr_cut"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if plot_param["mr_cut"] is not None:
            mag_r = self._get_quantity(catalog_instance, "mag_r",
                                       redshift_limit=redshift_limit,
                                       redshift_block_limit=redshift_block_limit)
            slct = slct & (mag_r < plot_param["mr_cut"])
            title += "mr < {}, ".format(plot_param["mr_cut"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if plot_param["stellar_mass_cut"] is not None:
            sm = self._get_quantity(catalog_instance, "stellar_mass",
                                    redshift_limit=redshift_limit,
                                    redshift_block_limit=redshift_block_limit)
            slct = slct & (np.log10(sm) > plot_param["stellar_mass_cut"])
            title += "$\\log_{{10}}(M_{{*}}/M_\\odot) > {}$, ".format(plot_param["stellar_mass_cut"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"
        if plot_param["halo_mass_cut"] is not None:
            halo_mass = self._get_quantity(catalog_instance, "halo_mass",
                                           redshift_limit=redshift_limit,
                                           redshift_block_limit=redshift_block_limit)
            slct = slct & (np.log10(halo_mass) > plot_param["halo_mass_cut"])
            title += "$\\log_{{10}}(M_{{halo}}/M_\\odot) > {}$, ".format(plot_param["halo_mass_cut"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if plot_param["synthetic_type"] is not None:
            upid = self._get_quantity(catalog_instance, "baseDC2/upid",
                                      redshift_limit=redshift_limit,
                                      redshift_block_limit=redshift_block_limit)
            slct = slct & (upid == plot_param["synthetic_type"])
            title += "synth = {}, ".format(plot_param["synthetic_type"])
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"

        if plot_param["red_sequence_cut"] is not None:
            rs = self._get_quantity(catalog_instance, "baseDC2/is_on_red_sequence_gr",
                                    redshift_limit=redshift_limit,
                                    redshift_block_limit=redshift_block_limit)
            slct = slct & (rs == plot_param["red_sequence_cut"])
            if plot_param["red_sequence_cut"]:
                title += "red sequence galaxies, "
            title_elem += 1
            if title_elem % title_elem_per_line == 0:
                title += "\n"
        #remove trailing ", "        
        title = title[0:-2]
        
        return slct, title


    def conclude_test(self, output_dir):
        # self.post_process_plot(self.summary_ax)
        # self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        # plt.close(self.summary_fig)
        pass
