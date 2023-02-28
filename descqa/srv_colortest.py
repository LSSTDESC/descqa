from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne
from scipy.stats import norm, binned_statistic
from mpi4py import MPI
import time

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


    def plot_mag_perband(self, mags, output_dir):
        ''' make per band and per model plots  '''
        # make flags on all bands and with flags
        masks = {}
        for model in self.models:
            mask_init = mags['mag_r_'+model]<=self.mag_lim
            for band in self.bands:
                mask_init = mask_init&(mags['mag_'+band+'_'+model]<=self.mag_lim)
                #mask_init = mask_init&(mags[model+'Flux_flag_'+band]==False)
                mask_init = mask_init&(mags['snr_'+band+'_'+model]>=self.snr_lim) 
            masks[model] = mask_init
     
        for band in self.bands:
            plt.figure()
            plt.title('mag_'+band)
            for model in self.models:
                mask_band = mags['mag_'+band+'_'+model]<self.mag_lim
                mask_band = mask_band & (mags['snr_'+band+'_'+model]>=self.snr_lim)
                mag_vals = mags['mag_'+band+'_'+model][mask_band]#[masks[model]]
                plt.hist(mag_vals,bins=np.linspace(22,self.mag_lim,100),alpha=0.5,label=model)
            plt.legend()
            plt.xlabel('mag_'+band)
            plt.savefig(os.path.join(output_dir,'mag_'+band+'.png'))
            plt.close()

        return 
    


    def plot_color_permodel(self,mags,output_dir):
        ''' '''

        # color plots for each model
        plt.figure(figsize=(8,8))
        for model in self.models:
            mask_init = (mags['mag_r_'+model]<=self.mag_lim)&(mags['mag_g_'+model]<=self.mag_lim)
            g_min_r = mags['mag_g_'+model][mask_init] - mags['mag_r_'+model][mask_init]
            plt.hist(g_min_r, bins = np.linspace(-2,2,100),label=model, alpha=0.5)
        plt.legend()
        plt.xlabel('g_min_r')
        plt.savefig(os.path.join(output_dir,'gmr.png'))
        plt.close()

        # color-color plots
        for model in self.models:
            plt.figure(figsize=(8,8))
            mask_init = (mags['mag_r_'+model]<=self.mag_lim)&(mags['mag_g_'+model]<=self.mag_lim)
            mask_init = (mags['mag_i_'+model]<=self.mag_lim)
            g_min_r = mags['mag_g_'+model][mask_init] - mags['mag_r_'+model][mask_init]
            r_min_i = mags['mag_r_'+model][mask_init] - mags['mag_i_'+model][mask_init]
            plt.hist2d(g_min_r, r_min_i, bins = np.linspace(-0.5,1.5,100))
            plt.title(model)
            plt.xlabel('g_min_r')
            plt.ylabel('r_min_i')
            plt.savefig(os.path.join(output_dir,'gmr_rmini_'+model+'.png'))
            plt.close()

        # comparison plots for color
        model1 = self.models[0]
        mask1 = (mags['mag_r_'+model1]<=self.mag_lim)&(mags['mag_g_'+model1]<=self.mag_lim)
        for model in self.models[1:]:
            mask2 = (mags['mag_r_'+model]<=self.mag_lim)&(mags['mag_g_'+model]<=self.mag_lim)
            mask_tot = mask1&mask2
            g_min_r1 = mags['mag_g_'+model1][mask_tot] - mags['mag_r_'+model1][mask_tot]
            g_min_r2 = mags['mag_g_'+model][mask_tot] - mags['mag_r_'+model][mask_tot]
            plt.figure(figsize=(8,8))
            plt.scatter(g_min_r1,g_min_r2-g_min_r1,s=0.01)
            a1,b1,c1 = binned_statistic(g_min_r1,g_min_r2-g_min_r1,bins= np.linspace(-0.5,1.5,100),statistic='median')
            plt.plot((b1[1:]+b1[:-1])/2.,a1,'r--')
            plt.plot((b1[1:]+b1[:-1])/2.,np.zeros_like(a1),'k')
            plt.xlabel(model1)
            plt.ylabel(model+'-'+model1)
            plt.xlim([-0.5,1.5])
            plt.ylim([-2,2])
            plt.savefig(os.path.join(output_dir,'gmr'+model1+'_'+model+'.png'))
            plt.close()

        # comparison plots for mag 
        model1 = self.models[0]
        for band in self.bands:
            mask1 = (mags['mag_'+band+'_'+model1]<=self.mag_lim)&(mags['snr_'+band+'_'+model1]>=self.snr_lim)
            #mask_init = mask_init&(mags[model+'Flux_flag_'+band]==False)
            for model in self.models[1:]:
                mask2 = (mags['mag_'+band+'_'+model]<=self.mag_lim)&(mags['snr_'+band+'_'+model]>=self.snr_lim)
                mask_tot = mask1&mask2
                g_min_r1 = mags['mag_'+band+'_'+model1][mask_tot]
                g_min_r2 = mags['mag_'+band+'_'+model][mask_tot]
                plt.figure(figsize=(8,8))
                plt.scatter(g_min_r1,g_min_r2-g_min_r1,s=0.01)
                a1,b1,c1 = binned_statistic(g_min_r1,g_min_r2-g_min_r1,bins= np.linspace(18,self.mag_lim,100),statistic='median')
                plt.plot((b1[1:]+b1[:-1])/2.,a1,'r--')
                plt.plot((b1[1:]+b1[:-1])/2.,np.zeros_like(a1),'k')
                plt.ylim([-1.5,1.5])
                plt.xlim([18,self.mag_lim])
                plt.xlabel(model1)
                plt.ylabel(model +' - '+ model1)
                plt.title('mag_'+band)
                plt.savefig(os.path.join(output_dir,'mag_'+band+'_'+model1+'_'+model+'_scatter.png'))
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
                print(model)
                quantities.append(model+'Flux_'+band); quantities.append(model+'FluxErr_'+band); quantities.append(model+'Flux_flag_'+band) #fluxes
                quantities.append('mag_'+band + '_'+model); quantities.append('magerr_'+band+'_'+model); quantities.append('snr_'+band+'_'+model); #mags
        quantities = tuple(quantities)
        # note that snr is defined on flux directly and not on magnitudes

        # reading in the data 
        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities,filters=filters,return_iterator=False)
        else:
            catalog_data = catalog_instance.get_quantities(quantities,return_iterator=False)
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

        self.plot_mag_perband(recvbuf, output_dir)
        self.plot_color_permodel(recvbuf, output_dir)



        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
