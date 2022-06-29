from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne
from scipy.stats import norm
from mpi4py import MPI
import healpy as hp
import time

from .base import BaseValidationTest, TestResult
from .plotting import plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckFluxes']


class CheckFluxes(BaseValidationTest):
    """
    Check of number of galaxies given flags and filters
    """

    def __init__(self, **kwargs):
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.lgndtitle_fontsize = kwargs.get('lgndtitle_fontsize', 12)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.no_version = kwargs.get('no_version', False)
        self.title_size = kwargs.get('title_size', 'small')
        self.font_size	= kwargs.get('font_size', 12)
        self.legend_size = kwargs.get('legend_size', 'x-small')
        self.ra = kwargs.get('ra')
        self.dec = kwargs.get('dec')

        self.cmodel = kwargs.get('cmodel')
        self.cmodel_err = kwargs.get('cmodel_err')
        self.cmodel_flag = kwargs.get('cmodel_flag')
        self.bands = kwargs.get('bands',[])

        if not any((
                self.bands
        )):
            raise ValueError('must specify bands')


        self.enable_individual_summary = bool(kwargs.get('enable_individual_summary', True))
        self.enable_aggregated_summary = bool(kwargs.get('enable_aggregated_summary', False))
        self.always_show_plot = bool(kwargs.get('always_show_plot', True))

        self.nbins = int(kwargs.get('nbins', 50))
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
            self.record_result('Running galaxy number test on {} {}'.format(
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
        for i, band in enumerate(self.bands):
            bands = band['bands']

        print(filters)
        lgnd_loc_dflt ='best'


        flags_tot =[]
        label_tot=[]
        plots_tot=[]
        quantities_new = []

        for band in bands:
            quantity = self.cmodel+band
            quantity_err = self.cmodel_err + band
            quantities_new.append(quantity)
            quantities_new.append(quantity_err)

        quantities_new = tuple(quantities_new)


        # reading in the data 
        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,filters=filters,return_iterator=False, rank=rank, size=size)
        else:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,return_iterator=False, rank=rank, size=size)
        a = time.time()


        for i, checks in enumerate(self.flags_to_check):
            #quantities_this = checks['quantities']
            kind = checks['kind']
            flag_val = checks['flag_val']
            quantities_this = flags_tot[i]

            #        for quantities_this in flags_tot:
            fig = None; ax=None;
            if rank==0:
                fig, ax = plt.subplots()
            has_plot = False

            for quantity in quantities_this:
                #PL : only currently works for doubles 
                value = catalog_data[quantity] 
                count = len(value)
                tot_num = comm.reduce(count)
                counts = comm.allgather(count)
                if rank==0:
                    if kind=='double':
                        recvbuf = np.zeros(tot_num)
                    elif kind=='bool':
                        recvbuf = np.zeros(tot_num)!=0.0
                else:
                    recvbuf = None
                displs = np.array([sum(counts[:p]) for p in range(size)])
                if kind=='double':
                    comm.Gatherv([value,MPI.DOUBLE], [recvbuf,counts,displs,MPI.DOUBLE],root=0)
                elif kind=='float':
                    comm.Gatherv([value,MPI.FLOAT], [recvbuf,counts,displs,MPI.FLOAT],root=0)
                elif kind=='int':
                    comm.Gatherv([value,MPI.INT], [recvbuf,counts,displs,MPI.INT],root=0)
                elif kind=='bool':
                    comm.Gatherv([value,MPI.BOOL], [recvbuf,counts,displs,MPI.BOOL],root=0)
                elif kind=='int64':
                    comm.Gatherv([value,MPI.INT64_T], [recvbuf, counts, displs, MPI.INT64_T],root=0)
                else:
                    print("add proper exception catch here")

                need_plot = False

                if rank==0:
                    print(time.time()-a,'read data',rank)

                a = time.time()
                
                if rank==0:
                    galaxy_count = len(recvbuf)
                    self.record_result('Found {} entries in this catalog.'.format(galaxy_count))
                    print(quantity)
                    print(flag_val)
                    print(recvbuf[0])
                    print('ngals masked = ',np.sum(recvbuf),' ngals unmasked = ', len(recvbuf))

                    need_plot=True


                    print(time.time()-a,'found entries')
                    a = time.time()

                    result_this_quantity = (
                             np.sum(recvbuf),
                            ('fail' if len(recvbuf)>(np.sum(recvbuf)*2.0) else 'pass'),
                        )
                    quantity_hashes[result_this_quantity[0]].add(quantity)
                    self.record_result(
                        result_this_quantity,
                        quantity ,
                        plots_tot[i]
                    )
                    
                if (need_plot or self.always_show_plot) and rank==0:
                    #    # PL: Changed - need to change this to a numpy function and then communicate it before plotting
                    xbins = np.linspace(np.min(recvbuf_ra),np.max(recvbuf_ra),50)
                    ybins = np.linspace(np.min(recvbuf_dec),np.max(recvbuf_dec),50)
                    area = (xbins[1]-xbins[0])*(ybins[1]-ybins[0])*(60.**2) #arcminutes
                    im = ax.hist2d(recvbuf_ra[recvbuf],recvbuf_dec[recvbuf], bins=(xbins,ybins),weights = 1./area*np.ones(len(recvbuf_ra[recvbuf])))#, label=quantity)
                    has_plot = True
                    b = time.time()
            
                if has_plot and (rank==0):
                    #ax.set_xlabel(('log ' if checks.get('log') else '') + quantity_group_label, size=self.font_size)
                    #ax.yaxis.set_ticklabels([])
                    if checks.get('plot_min') is not None: #zero values fail otherwise
                       ax.set_xlim(left=checks.get('plot_min'))
                    if checks.get('plot_max') is not None:
                        ax.set_xlim(right=checks.get('plot_max'))
                    ax.set_title('{} {}'.format(catalog_name, version), fontsize=self.title_size)
                    fig.colorbar(im[3], ax=ax)
                    ax.colorbar=True
                    fig.tight_layout()
                    print(plots_tot[i])
                    print("closing figure")
                    fig.savefig(os.path.join(output_dir, plots_tot[i]))
                    plt.close(fig)

        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
