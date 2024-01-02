from __future__ import unicode_literals, absolute_import, division
import os
import re
import numpy as np
from scipy.interpolate import interp1d
from GCR import GCRQuery
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from .plotting import plt
from .base import BaseValidationTest, TestResult

possible_observations = {
    'SDSS': {
        'filename_template': 'AGNdata/SDSS/richards_2006_table2.dat',
        'usecols': {'g': {'N<m':(0, 3, 4, 5),
                          'dN/dm':(0, 1, 2, 5),
                         },
                    'i': {'N<m':(0, 8, 9, 10),
                          'dN/dm':(0, 6, 7, 10),
                         },
                   },
        'colnames': ('mag', 'n', 'err', 'N_Q'),
        'skiprows': 5,
        'missing_values':'...',
        'bin-width': 0.25,
        'label': 'Richards et. al. (2006)\nSDSS',
        'color': 'r',
        'title': '$\\rm M_{}<{}$, $\\rm {}<z<{}$',
        'ytitle':{'dN/dm':'$\\rm N (deg^{{-2}} {}mag^{{-1}})$',
                  'N<m':'$\\rm N(<{}) (deg^{{-2}})$',
                 },
    },
}

__all__ = ['AGN_NumberDensity']


class AGN_NumberDensity(BaseValidationTest):
    """
    AGN number desnsity test
    """
    def __init__(self, band='g', rest_frame_band='i', Mag_lim=-22.5, z_lim=(0.4, 2.1),
                 observation='SDSS', **kwargs):
        """
        parameters
        ----------
        band : string
            photometric band

        rest_frame_band : string
            photometric band

        Mag_lim : float
            absolute magnitude upper limits

        z_lim: list
            lower and upper limits of redshifts to select

        observation : string
            string indicating which obsrvational data to use for validating
        """
        # pylint: disable=super-init-not-called
        # pylint: disable=too-many-instance-attributes

        self.no_agn_extinction = kwargs.get('no_agn_extinction', True)        
        # catalog quantities needed
        noagnext = '' if self.no_agn_extinction else 'extincted_agn_'
        possible_mag_fields = ('mag_{}_{}sdss',
                               'mag_{}_{}lsst',
                              )
        self.possible_mag_fields = [f.format(band, noagnext) for f in possible_mag_fields]

        # cut on host galaxy + AGN flux 
        possible_Mag_fields = ('Mag_true_{}_{}sdss_z0',
                               'Mag_true_{}_{}lsst_z0',
                              )
        self.possible_Mag_fields = [f.format(rest_frame_band, noagnext) for f in possible_Mag_fields]
        
        # for fraction cut
        possible_agn_mag_fields = ('mag_{}_agnonly_{}sdss',
                                     'mag_{}_agnonly_{}lsst',
                                    )
        self.possible_agn_mag_fields = [f.format(band, noagnext) for f in possible_agn_mag_fields] 
        self.duty_cycle_quantity = 'duty_cycle_on'
        
        # attach some attributes to the test
        self.band = band
        self.rest_frame_band = rest_frame_band
        self.Mag_lim = Mag_lim
        self.z_lim = list(z_lim)
        self.duty_cycle_on = kwargs.get('duty_cycle_on', True)
        self.agn_flux_fraction = kwargs.get('agn_flux_fraction', 0.)
        self.font_size = kwargs.get('font_size', 20)
        self.title_size = kwargs.get('title_size', 20)
        self.legend_size = kwargs.get('legend_size', 18)
        self.no_title = kwargs.get('no_title', False)
        self.no_legend_title = kwargs.get('no_legend_title', False)
        self.truncate_legend_title = kwargs.get('truncate_legend', True)
        self.nrows = kwargs.get('nrows', 1)
        self.cumulative_only = kwargs.get('cumulative_only', False)
        self.ncolumns = 1 if self.cumulative_only else kwargs.get('ncolumns', 2)
        self.mag_lo = kwargs.get('mag_lo', 14)
        self.mag_hi = kwargs.get('mag_hi', 22)
        self.validation_range = kwargs.get('validation_range', (16., 19.))
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.figx_p = kwargs.get('figx_p', 7)
        self.figy_p = kwargs.get('figy_p', 7)
        self.msize = kwargs.get('msize', 6)
        
        # set color of lines in plots
        colors = plt.cm.jet(np.linspace(0, 1, 2)) # pylint: disable=no-member
        if band == 'g':
            self.line_color = colors[0]
        elif band == 'i':
            self.line_color = colors[1]
        else:
            self.line_color = 'black'

        # check for validation observation
        if not observation:
            print('Warning: no data file supplied, no observation requested; only catalog data will be shown.')
        elif observation not in possible_observations:
            raise ValueError('Observation: {} not available for this test.'.format(observation))
        else:
            self.validation_data = self.get_validation_data(band, observation)

        # prepare summary plot
        self.summary_fig, self.summary_axs = self.setup_subplots()
        self.first_pass = True

    def get_validation_data(self, band, observation):
        """
        load (observational) data to use for validation test
        """
        data_args = possible_observations[observation]
        data_path = os.path.join(self.data_dir, data_args['filename_template'].format(band))


        if not os.path.exists(data_path):
            raise ValueError("{}-band data file {} not found".format(band, data_path))

        if not os.path.getsize(data_path):
            raise ValueError("{}-band data file {} is empty".format(band, data_path))

        save_keys = [k for k in data_args.keys() if 'cols' not in k and 'file' not in k and 'skip' not in k]
        validation_data = dict(zip(save_keys, [data_args[k] for k in save_keys]))
        validation_data['data'] = {}

        for k, v in data_args['usecols'][band].items():
            data = np.genfromtxt(data_path, unpack=True, usecols=v, skip_header=data_args['skiprows'],
                                 missing_values=data_args['missing_values'])
            validation_data['data'][k] = dict(zip(data_args['colnames'], data))

        return validation_data


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        """
        """

        mag_field_key = catalog_instance.first_available(*self.possible_mag_fields)
        Mag_field_key = catalog_instance.first_available(*self.possible_Mag_fields)
        mag_agn_key = catalog_instance.first_available(*self.possible_agn_mag_fields)
        duty_cycle_key = catalog_instance.first_available(self.duty_cycle_quantity)
        if not mag_field_key:
            return TestResult(skipped=True,
                              summary='Catalog is missing requested quantity: {}'.format(self.possible_mag_fields))
        if not Mag_field_key:
            return TestResult(skipped=True,
                              summary='Catalog is missing requested quantity: {}'.format(self.possible_Mag_fields))
        if not mag_agn_key:
            return TestResult(skipped=True,
                              summary='Catalog is missing requested quantity: {}'.format(self.possible_agn_mag_fields))
        if self.duty_cycle_on and not duty_cycle_key:
            return TestResult(skipped=True,
                              summary='Catalog is missing requested quantity: {}'.format(self.duty_cycle_quantity))
        
        # check to see if catalog is a light cone
        # this is required since we must be able to calculate the angular area
        if not catalog_instance.lightcone:
            return TestResult(skipped=True, summary="Catalog is not a light cone.")

        # check to see the angular area if an attribute of the catalog
        try:
            sky_area = catalog_instance.sky_area
        except AttributeError:
            return TestResult(skipped=True, summary="Catalog needs an attribute 'sky_area'.")

        filtername = mag_field_key.split('_')[(-1 if mag_field_key.startswith('m') else -2)].upper()  #extract filtername
        filelabel = '_'.join((filtername, self.band))

        z_filters = ['redshift > {}'.format(self.z_lim[0]), 'redshift < {}'.format(self.z_lim[1])]
        Mag_filters = ['{} < {}'.format(Mag_field_key, self.Mag_lim)]
        filters = z_filters + Mag_filters + [duty_cycle_key] if self.duty_cycle_on else z_filters + Mag_filters
        
        # retreive data from mock catalog
        catalog_data = catalog_instance.get_quantities([mag_field_key, mag_agn_key], filters=filters)
        d = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)
        
        #select point-like sources according to specified agn_flux_fraction
        N_all = len(d[mag_field_key])
        fluxid = 'AGN+galaxy'
        if self.agn_flux_fraction > 0. and self.agn_flux_fraction < 1.:
            point_like_mask = (d[mag_agn_key] - d[mag_field_key] < -2.5*np.log10(self.agn_flux_fraction))
            mags = d[mag_field_key][point_like_mask]
        elif self.agn_flux_fraction == 0.:
            mags = d[mag_field_key]
        else:
            mags = d[mag_agn_key]
            fluxid = 'AGN'
            
        #####################################################
        # caclulate the number densities of AGN
        #####################################################

        # get the total number of AGN passing cut and save for txt file
        N_tot = len(mags)
        if self.agn_flux_fraction > 0.:
            total_txt = '{}/{} with AGN magnitude fraction > {}'.format(N_tot, N_all, self.agn_flux_fraction)
            fraction_txt = '$\\rm F_{{AGN}}/F_{{Total}} > {}$'.format(self.agn_flux_fraction)
        else:
            total_txt = '{} AGN'.format(N_tot)
            fraction_txt = ''
        fraction_txt = '$\\rm F_{{AGN}}/F_{{Total}} > {}$'.format(self.agn_flux_fraction)
            
        # define the apparent magnitude bins for plotting purposes
        dmag = self.validation_data.get('bin_width', 0.25)

        mag_bins = np.arange(self.mag_lo, self.mag_hi + dmag, dmag)
        mag_cen = (mag_bins[:-1] + mag_bins[1:])/2

        # calculate differential binned data (validation data does not divide by bin width)
        Ndm, _ = np.histogram(mags, bins=mag_bins)
        dn = Ndm/sky_area

        # calculate N(<mag) in the bins
        N = np.cumsum(Ndm)
        n_cum = N/sky_area

        #################################################
        # plot the differential and cumulative apparent magnitude function
        #################################################

        fig, axs = self.setup_subplots()

        if self.truncate_cat_name:
            catalog_name = re.split('_', catalog_name)[0]
        results = {'catalog':{}, 'data':{}}
        colname = 'n'

        legend_title = ''
        if not self.no_legend_title:
            if self.truncate_legend_title:
                legend_title = 'Duty-cycle on'
            else:
                legend_title = '; '.join(('Duty-cycle on' if self.duty_cycle_on else 'No duty-cycle',
                                          fraction_txt,
                                          'No AGN ext.' if self.no_agn_extinction else 'AGN ext.'))
        
        if not self.cumulative_only:
            mag_pts_list = [mag_bins[1:], mag_cen]
            cat_data_list = [n_cum, dn]
            ytitles = [self.band, dmag]
            val_tuples = self.validation_data['data'].items()
        else:
            mag_pts_list = [mag_bins[1:]]
            cat_data_list = [n_cum]
            ytitles = [self.band]
            val_tuples = [ (k, v) for k, v in self.validation_data['data'].items() if 'N<m' in k]
        for ax, summary_ax, mag_pts, cat_data, ytit, (k, val_data) in zip(axs, self.summary_axs, 
                                                                          mag_pts_list,
                                                                          cat_data_list, ytitles,
                                                                          val_tuples):
            # plot
            ax[0].plot(mag_pts, cat_data, label=catalog_name, color=self.line_color)
            ax[0].errorbar(val_data['mag'], val_data['n'], yerr=val_data['err'], label=self.validation_data['label'],
                           color=self.validation_data['color'], ms=self.msize, fmt='o')
            summary_ax[0].plot(mag_pts, cat_data, label=catalog_name, color=self.line_color)
            if self.first_pass:
                summary_ax[0].errorbar(val_data['mag'], val_data['n'], yerr=val_data['err'],
                                       label=self.validation_data['label'],
                                       color=self.validation_data['color'], ms=self.msize, fmt='o')
                self.decorate_plot(summary_ax[0], self.validation_data['ytitle'][k].format(ytit), scale='log')

            #decorate
            self.decorate_plot(ax[0], self.validation_data['ytitle'][k].format(ytit), scale='log',
                               legend_title=legend_title)

            # get fractional diffrence between the mock catalog and validation data
            mag_val_pts, delta = self.get_frac_diff(mag_pts, cat_data, val_data['mag'], val_data['n'],
                                                    self.validation_range)
            ax[1].plot(mag_val_pts, delta, color=self.line_color, label='Frac. Diff.')
            summary_ax[1].plot(mag_val_pts, delta, color=self.line_color, label='Frac. Diff.')
            self.decorate_plot(ax[1], ylabel=r'$\Delta n/n$', scale='linear',
                               xlabel=r'$\rm m_{}^{{{}}}$'.format(self.band, fluxid),
                               legend_title=legend_title, legend_loc='lower right')
            if self.first_pass:
                self.decorate_plot(summary_ax[1], ylabel=r'$\Delta n/n$', scale='linear',
                                   xlabel='$\\rm m_{}^{{{}}}$'.format(self.band, fluxid))

            #save plotted points
            N_tot_data = np.sum(val_data['N_Q']) # total number of AGN
            results['catalog'][k] = {'mag': mag_pts, colname:cat_data}
            results['data'][k] = {'mag': val_data['mag'], colname:val_data['n'], colname + '+-':val_data['err']}

        if not self.no_title:
            fig.suptitle(self.validation_data['title'].format(self.rest_frame_band, self.Mag_lim,
                                                              self.z_lim[0], self.z_lim[1]),
                         fontsize=self.title_size, y=1.00)
            self.summary_fig.suptitle(self.validation_data['title'].format(self.rest_frame_band, self.Mag_lim,
                                                                           self.z_lim[0], self.z_lim[1]),
                                      fontsize=self.title_size, y=1.00)

        #save results for catalog and validation data in txt files
        for filename, dtype, ntot in zip_longest((filelabel, re.sub(' ', '_', self.validation_data['label'])),
                                                 ('catalog', 'data'),
                                                 (total_txt, N_tot_data)):
            if filename and dtype:
                with open(os.path.join(output_dir, 'Nagn_' + filename + '.txt'), 'ab') as f_handle:
                    for key, value in results[dtype].items():
                        self.save_quantities(colname, value, f_handle,
                                             comment=' '.join((key, ('#AGN = {}'.format(ntot) or ''))))

        if self.first_pass: #turn off validation data plot in summary for remaining catalogs
            self.first_pass = False

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, 'Nagn_vs_mag_{}.png'.format(self.band)))
        plt.close(fig)

        return TestResult(0, inspect_only=True)


    def setup_subplots(self):
        """
        """
        fig = plt.figure(figsize=(self.figx_p*self.ncolumns, self.figy_p))
        gs = fig.add_gridspec(2*self.nrows, self.ncolumns, height_ratios=[3, 1])
        axs = []
        axs.append([fig.add_subplot(gs[0])])
        gs_index = 1 if self.cumulative_only else 2
        axs[0].append(fig.add_subplot(gs[gs_index], sharex=axs[0][0]))
        if not self.cumulative_only:
            axs.append([fig.add_subplot(gs[1])])
            axs[1].append(fig.add_subplot(gs[3], sharex=axs[1][0]))

        return fig, axs


    @staticmethod
    def get_frac_diff(mag_pts, cat_data, val_mags, val_data, validation_range):
        """
        """
        mask = (val_data > 0)
        x = val_mags[mask]
        y = np.log10(val_data[mask])
        f = interp1d(x, y, fill_value='extrapolate')
        val_mask = (mag_pts > validation_range[0]) & (mag_pts < validation_range[1])
        interp_data = 10**f(mag_pts[val_mask])

        return mag_pts[val_mask], (cat_data[val_mask] - interp_data)/interp_data


    def decorate_plot(self, ax, ylabel, scale='log', xlabel=None, legend_title='', legend_loc='upper left'):
        """
        """
        ax.set_ylabel(ylabel, size=self.font_size)
        leg = ax.legend(loc=legend_loc, fancybox=True, framealpha=0.5,
                        fontsize=self.legend_size, numpoints=1)
        leg.set_title(legend_title, prop={'size':self.legend_size})
        ax.set_yscale(scale)
        if xlabel:
            ax.set_xlabel(xlabel, size=self.font_size)
        else:
            for axlabel in ax.get_xticklabels():
                axlabel.set_visible(False)


    @staticmethod
    def post_process_plot(fig):
        """
        """
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(top=0.94)

    @staticmethod
    def save_quantities(keyname, results, filename, comment=''):
        """
        """
        if keyname in results:
            if keyname+'-' in results and keyname+'+' in results:
                fields = ('mag', keyname, keyname+'-', keyname+'+')
                header = ', '.join(('Data columns are: <mag>', keyname, keyname+'-', keyname+'+'))
            elif keyname+'+-' in results:
                fields = ('mag', keyname, keyname+'+-')
                header = ', '.join(('Data columns are: <mag>', keyname, keyname+'+-'))
            else:
                fields = ('mag', keyname)
                header = ', '.join(('Data columns are: <mag>', keyname))
            np.savetxt(filename, np.vstack((results[k] for k in fields)).T, fmt='%12.4e', header=': '.join([comment, header]))


    def conclude_test(self, output_dir):
        """
        output_dir: output directory
        """
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        print('saved')
        plt.close(self.summary_fig)
