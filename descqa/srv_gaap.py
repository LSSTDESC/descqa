from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne
import math
from scipy.stats import norm, binned_statistic
from mpi4py import MPI
import time
import matplotlib.colors as mcolors

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .parallel import send_to_master

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckFluxes']


class CheckFluxes(BaseValidationTest):
    """
    Check flux values and magnitudes
    """

    def __init__(self, **kwargs):
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.lgndtitle_fontsize = kwargs.get('lgndtitle_fontsize', 12)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.no_version = kwargs.get('no_version', False)
        self.title_size = kwargs.get('title_size', 'small')
        self.font_size  = kwargs.get('font_size', 12)
        self.legend_size = kwargs.get('legend_size', 'x-small')
        self.ra = kwargs.get('ra')
        self.dec = kwargs.get('dec')
        self.compare = kwargs.get('compare',False)
        self.models = kwargs.get('models')
        self.bands = kwargs.get('bands')
        self.mag_lim = kwargs.get('mag_lim')
        self.snr_lim = kwargs.get('snr_lim',0)
        self.aps = kwargs.get('aps')
        self.ap_list = kwargs.get('ap_list')
        self.color_list = list(mcolors.TABLEAU_COLORS.keys())

        if not any((
                self.catalog_filters,
        )):
            raise ValueError('you need to specify catalog_filters for these checks, add a good flag if unsure')

        self.enable_individual_summary = bool(kwargs.get('enable_individual_summary', True))
        self.enable_aggregated_summary = bool(kwargs.get('enable_aggregated_summary', False))
        self.always_show_plot = bool(kwargs.get('always_show_plot', True))

        self.nbins = int(kwargs.get('nbins', 20))
        self.prop_cycle = None

        self.current_catalog_name = None
        self.current_failed_count = None
        self._aggregated_header = list()
        self._aggregated_table = list()
        self._individual_header = list()
        self._individual_table = list()

        super(CheckFluxes, self).__init__(**kwargs)

    def record_result(self, results, quantity_name=None, more_info=None, failed=None, individual_only=False):
        if isinstance(results, dict):
            self.current_failed_count += sum(1 for v in results.values() if v[1] == 'fail')
        elif failed:
            self.current_failed_count += 1

        if self.enable_individual_summary:
            if quantity_name is None:
                self._individual_header.append(self.format_result_header(results, failed))
            else:
                self._individual_table.append(self.format_result_row(results, quantity_name, more_info))

        if self.enable_aggregated_summary and not individual_only:
            if quantity_name is None:
                results = '{} {}'.format(self.current_catalog_name, results) if self.current_catalog_name else results
                self._aggregated_header.append(self.format_result_header(results, failed))
            else:
                quantity_name = '{} {}'.format(self.current_catalog_name, quantity_name) if self.current_catalog_name else quantity_name
                self._aggregated_table.append(self.format_result_row(results, quantity_name, more_info))

    def format_result_row(self, results, quantity_name, more_info):
        more_info = 'title="{}"'.format(more_info) if more_info else ''
        output = ['<tr>', '<td {1}>{0}</td>'.format(quantity_name, more_info)]
        output.append('</tr>')
        return ''.join(output)

    @staticmethod
    def format_result_header(results, failed=False):
        return '<span {1}>{0}</span>'.format(results, 'class="fail"' if failed else '')


    def plot_snr_ap(self, fluxes, output_dir):
        '''
        Plot the median SNR vs aperture width curve for each band, overplotting the SNR of the 
        optimal elliptical aperture. This plots the total number of galaxies above a SNR threshold, 
        the median SNR of those galaxies, and the sum of the SNR values. You expect the optimal aperture 
        to do well compared to the circular apertures for these metrics, and the total SNR to be above 
        or at that of the circular apertures. 
        '''
        fig= plt.figure(figsize = (8,24))
        plt.subplot(311)
        plt.title('median SNR')
        plt.subplot(312)
        plt.title('ngals above SNR')
        plt.subplot(313)
        plt.title('total SNR')

        i = 0 
        for band in self.bands:
            n_ap=[]
            snr_ap=[]
            snr_tot=[]
            for ap in self.aps:
                mask_ap = (fluxes['gaapFlux_flag_'+ap+'_'+band]==0)&(fluxes['snr_'+ap+'_'+band+'_gaap']>=self.snr_lim)
                snr = fluxes['snr_'+ap+'_'+band+'_gaap'][mask_ap]
                n_ap.append(len(snr))
                snr_ap.append(np.median(snr))
                snr_tot.append(np.sum(snr))
            plt.subplot(311)
            plt.plot(self.ap_list,snr_ap,label=band, c=self.color_list[i])
            mask_optimal = (fluxes['gaapFlux_flag_'+band]==0)&(fluxes['snr_'+band+'_gaap']>=self.snr_lim)
            snr_optimal = fluxes['snr_'+band+'_gaap'][mask_optimal]
            plt.plot(self.ap_list,np.median(snr_optimal)*np.ones(len(snr_ap)),'--', c=self.color_list[i])
            plt.subplot(312)
            plt.plot(self.ap_list,n_ap,label=band, c = self.color_list[i])
            plt.plot(self.ap_list,len(snr_optimal)*np.ones(len(snr_ap)),'--', c = self.color_list[i])
            plt.subplot(313)
            plt.plot(self.ap_list,snr_tot,label=band, c=self.color_list[i])
            plt.plot(self.ap_list,np.sum(snr_optimal)*np.ones(len(snr_ap)),'--', c=self.color_list[i])
            i += 1
        plt.xlabel('aperture')
        plt.subplot(311)
        plt.legend(loc = 'upper center')

        for ax in fig.get_axes():
            ax.label_outer()

        plt.savefig(os.path.join(output_dir,'snr_ap_'+band+'.png'))
        plt.close()



        return 

    def plot_gi_ap(self, fluxes, output_dir):
        '''
        compare g-i color in circular apertures to that of the optimal aperture. You expect this to match at around the peak of the SNR 
        curve in the above function, and diverge at large apertures where the outskirts of the galaxy contribute to the color. 
        '''
        fig, axs = plt.subplots(2,3,figsize=(12,8))
        fig.suptitle('g - i aperture vs optimal')
        count_plot = 0 

        for ap in self.aps:
            idx1 = count_plot%2
            idx2 = math.floor(count_plot/2)

            mask_ap = (fluxes['snr_'+ap+'_g_gaap']>=self.snr_lim)& (fluxes['snr_'+ap+'_i_gaap']>=self.snr_lim)&(fluxes['snr_g_gaap']>=self.snr_lim)&(fluxes['snr_i_gaap']>=self.snr_lim) # passes SNR lim in both
            g_min_i = fluxes['mag_'+ap+'_g_gaap'][mask_ap] - fluxes['mag_'+ap+'_i_gaap'][mask_ap]
            g_min_i_opt = fluxes['mag_g_gaap'][mask_ap] - fluxes['mag_i_gaap'][mask_ap]
            axs[idx1,idx2].hist2d(g_min_i_opt,g_min_i,bins=np.linspace(-1,3,100))
            axs[idx1,idx2].set_title(ap)
            axs[idx1,idx2].set_xlabel('g_min_i optimal')
            axs[idx1,idx2].set_ylabel('g_min_i aperture ')
            count_plot +=1 

        for ax in fig.get_axes():
            ax.label_outer()

        plt.savefig(os.path.join(output_dir,'g_min_i_ap.png'))
        plt.close()

        return 

    def plot_ri_gr_ap(self, fluxes, output_dir):
        '''
        Compare r-i / g-r color-color plot for different circular apertures and the optimal aperture. You expect to see for DC2 a 
        wishbone-type image with distinct populations. At larger apertures this may get blurred as the signal to noise decreases. 
        '''

        fig, axs = plt.subplots(2,3,figsize=(12,8))
        fig.suptitle('r - i vs g - r for apertures')
        count_plot = 0
        for ap in self.aps:
            idx1 = count_plot%2
            idx2 = math.floor(count_plot/2)

            mask_ap = (fluxes['snr_'+ap+'_g_gaap']>=self.snr_lim)&(fluxes['snr_'+ap+'_r_gaap']>=self.snr_lim)& (fluxes['snr_'+ap+'_i_gaap']>=self.snr_lim)&(fluxes['snr_g_gaap']>=self.snr_lim)&(fluxes['snr_i_gaap']>=self.snr_lim)&(fluxes['snr_r_gaap']>=self.snr_lim) # passes SNR lim in both
            g_min_r = fluxes['mag_'+ap+'_g_gaap'][mask_ap] - fluxes['mag_'+ap+'_r_gaap'][mask_ap]
            g_min_r_opt = fluxes['mag_g_gaap'][mask_ap] - fluxes['mag_r_gaap'][mask_ap]
            r_min_i = fluxes['mag_'+ap+'_r_gaap'][mask_ap] - fluxes['mag_'+ap+'_i_gaap'][mask_ap]
            r_min_i_opt = fluxes['mag_r_gaap'][mask_ap] - fluxes['mag_i_gaap'][mask_ap]


            axs[idx1,idx2].hist2d(g_min_r,r_min_i,bins=np.linspace(-1,3,100))
            axs[idx1,idx2].set_title(ap)
            axs[idx1,idx2].set_xlabel('g - r' )
            axs[idx1,idx2].set_ylabel('r - i')
            count_plot +=1 

        for ax in fig.get_axes():
            ax.label_outer()

        plt.savefig(os.path.join(output_dir,'rmi_gmr_ap.png'))
        plt.close()

        plt.figure(figsize=(8,8))
        plt.hist2d(g_min_r_opt,r_min_i_opt,bins=np.linspace(-1,3,100))
        plt.title('r - i vs g - r for optimal aperture')
        plt.xlabel('g - r' )
        plt.ylabel('r - i')
        plt.savefig(os.path.join(output_dir,'gmr_rmi_opt.png'))
        plt.close()

        return


    def plot_unresolved(self, fluxes, output_dir):
        '''
        Plotting unresolved sources, i.e. ones where the lowest apertures detect the same total magnitude. This 
        suggests stellar contamination in the sample.  
        '''
        mag_r_ap05 = fluxes['mag_0p5_r_gaap']
        mag_r_ap07 = fluxes['mag_0p7_r_gaap']
        plt.figure(figsize = (8,8))
        plt.title('unresolved sources in r-band')
        plt.plot(np.linspace(15,26,100),np.zeros(100),'k--')
        plt.scatter(mag_r_ap05,mag_r_ap05-mag_r_ap07,s=0.1)
        plt.xlim([15,26])
        plt.ylim([-0.05,0.1])
        plt.xlabel('m_0p5')
        plt.ylabel('m_0p5-m_0p7')
        plt.savefig(os.path.join(output_dir,'unresolved.png'))
        plt.close()
        return

        


    def generate_summary(self, output_dir, aggregated=False):
        if aggregated:
            if not self.enable_aggregated_summary:
                return
            header = self._aggregated_header
            table = self._aggregated_table
        else:
            if not self.enable_individual_summary:
                return
            header = self._individual_header
            table = self._individual_table

        with open(os.path.join(output_dir, 'SUMMARY.html'), 'w') as f:
            f.write('<html><head><style>html{font-family: monospace;} table{border-spacing: 0;} thead,tr:nth-child(even){background: #ddd;} thead{font-weight: bold;} td{padding: 2px 8px;} .fail{color: #F00;} .none{color: #444;}</style></head><body>\n')

            f.write('<ul>\n')
            for line in header:
                f.write('<li>')
                f.write(line)
                f.write('</li>\n')
            f.write('</ul><br>\n')

            f.write('<table><thead><tr><td>Quantity</td>\n')
            f.write('</tr></thead><tbody>\n')
            for line in table:
                f.write(line)
                f.write('\n')
            f.write('</tbody></table></body></html>\n')

        if not aggregated:
            self._individual_header.clear()
            self._individual_table.clear()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        all_quantities = sorted(map(str, catalog_instance.list_all_quantities(True)))

        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))
        self.current_catalog_name = catalog_name
        self.current_failed_count = 0
        galaxy_count = None
        quantity_hashes = defaultdict(set)

        if rank==0:
            self.record_result('Running flux and magnitude test on {} {}'.format(
                catalog_name,
                getattr(catalog_instance, 'version', ''),
                individual_only=True,
            ))

        if self.truncate_cat_name:
            catalog_name = catalog_name.partition("_")[0]
        version = getattr(catalog_instance, 'version', '') if not self.no_version else ''

        # create filter labels
        filters=[]
        for i, filt in enumerate(self.catalog_filters):
            filters = filt['filters']

        print(filters)
        lgnd_loc_dflt ='best'

        label_tot=[]
        plots_tot=[]

        # doing everything together this time so we can combine flags
        quantities = []
        print(type(self.models))
        print(type(self.bands))
        print(self.models[0])
        for band in self.bands:
            print(band)
            for model in self.models:
                # optimal ones
                quantities.append(model+'Flux_'+band); quantities.append(model+'FluxErr_'+band); quantities.append(model+'Flux_flag_'+band);
                quantities.append('mag_'+band + '_'+model); quantities.append('magerr_'+band+'_'+model);  #mags
                quantities.append('snr_'+band+'_'+model);
                for ap in self.aps:
                    quantities.append(model+'Flux_'+ap+'_'+band); quantities.append(model+'FluxErr_'+ap+'_'+band); quantities.append(model+'Flux_flag_'+ap+'_'+band);
                    quantities.append('mag_'+ap+'_'+band + '_'+model); quantities.append('magerr_'+ap+'_'+band+'_'+model);  #mags
                    quantities.append('snr_'+ap+'_'+band+'_'+model);
                    

        quantities = tuple(quantities)

        # reading in the data 
        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities,filters=filters,return_iterator=False, rank=rank, size=size)
        else:
            catalog_data = catalog_instance.get_quantities(quantities,return_iterator=False, rank=rank, size=size)
        a = time.time()

        
        data_rank={}
        recvbuf={}
        for quantity in quantities:
            data_rank[quantity] = catalog_data[quantity]
            print(len(data_rank[quantity]))
            if 'flag' in quantity:
                recvbuf[quantity] = send_to_master(data_rank[quantity],'bool')
            else:
                recvbuf[quantity] = send_to_master(data_rank[quantity],'double')

        if rank==0:
            print(len(recvbuf[quantity]))

        self.plot_snr_ap(recvbuf, output_dir)
        self.plot_gi_ap(recvbuf, output_dir)
        self.plot_unresolved(recvbuf, output_dir)
        self.plot_ri_gr_ap(recvbuf, output_dir)


        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
