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

from .base import BaseValidationTest, TestResult
from .plotting import plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckNgals']


class CheckNgals(BaseValidationTest):
    """
    Check of number of galaxies given flags and filters
    """

    def __init__(self, **kwargs):
        self.flags_to_check = kwargs.get('flags_to_check', [])
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.lgndtitle_fontsize = kwargs.get('lgndtitle_fontsize', 12)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.no_version = kwargs.get('no_version', False)
        self.title_size = kwargs.get('title_size', 'small')
        self.font_size	= kwargs.get('font_size', 12)
        self.legend_size = kwargs.get('legend_size', 'x-small')
        self.ra = kwargs.get('ra')
        self.dec = kwargs.get('dec')
        self.nside = kwargs.get('nside')

        if not any((
                self.flags_to_check,
                self.catalog_filters,
        )):
            raise ValueError('must specify flags_to_check, catalog_filters')


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

        super(CheckNgals, self).__init__(**kwargs)

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
        for s in range(2):
            output.append('<td class="{1}" title="{2}">{0:.4g}</td>'.format(*results[s]))
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
            for s in ['ngals (per sq arcmin)', 'percentage retained']:
                f.write('<td>{}</td>'.format(s))
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

        lgnd_loc_dflt ='best'


        flags_tot =[]
        label_tot=[]
        plots_tot=[]
        for i, checks in enumerate(self.flags_to_check):
            # total list of quantities 
            quantity_patterns = checks['quantities'] if isinstance(checks['quantities'], (tuple, list)) else [checks['quantities']]

            quantities_this = set()
            quantity_pattern = None
            for quantity_pattern in quantity_patterns:
                quantities_this.update(fnmatch.filter(all_quantities, quantity_pattern))

            if not quantities_this:
                if rank==0:
                    self.record_result('Found no matching quantities for {}'.format(quantity_pattern), failed=True)
                continue

            quantities_this = sorted(quantities_this)
            flags_tot.append(quantities_this)

            if 'label' in checks:
                quantity_group_label = checks['label']
            else:
                quantity_group_label = re.sub('_+', '_', re.sub(r'\W+', '_', quantity_pattern)).strip('_')
            plot_filename = 'p{:02d}_{}.png'.format(i, quantity_group_label)
            label_tot.append(quantity_group_label)
            plots_tot.append(plot_filename)


        quantities_this_new=[]
        for q in flags_tot:
            print(q)
            if len(q)>1:
                for j in q:
                    quantities_this_new.append(j)
            else:
                quantities_this_new.append(q[0])
        quantities_this_new.append(self.ra)
        quantities_this_new.append(self.dec)
        quantities_this_new = tuple(quantities_this_new)
        print(quantities_this_new)


        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,filters=filters,return_iterator=False, rank=rank, size=size)
        else:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,return_iterator=False, rank=rank, size=size)

        data_rank={}
        recvbuf={}
        for quantity in [self.ra,self.dec]:
            data_rank[quantity] = catalog_data[quantity]
            count = len(data_rank[quantity])
            tot_num = comm.reduce(count)
            counts = comm.allgather(count)
            if rank==0:
                recvbuf[quantity] = np.zeros(tot_num)
            else:
                recvbuf[quantity] = None
            displs = np.array([sum(counts[:p]) for p in range(size)])
            comm.Gatherv([data_rank[quantity],MPI.DOUBLE], [recvbuf[quantity], counts, displs, MPI.DOUBLE],root=0)
        recvbuf_ra = recvbuf[self.ra]
        recvbuf_dec = recvbuf[self.dec]

        if rank==0:
            galaxy_count = len(recvbuf_ra)
            self.record_result('Found {} entries in this catalog.'.format(galaxy_count))


        #idx_vals = np.arange(len(recvbuf_ra))
        #np.random.shuffle(idx_vals)
        #for i in range(10000):#catalog_instance.get_quantities(['ra_true', 'dec_true'], return_iterator=True):
        #    pixels.update(hp.ang2pix(self.nside, recvbuf_ra[idx_vals[i]], recvbuf_dec[idx_vals[i]], lonlat=True))
        #frac = len(pixels) / hp.nside2npix(self.nside)
        #skyarea = frac * np.rad2deg(np.rad2deg(4.0*np.pi))

        #print(skyarea)

        #hp_map = np.empty(hp.nside2npix(self.nside))
        #hp_map.fill(hp.UNSEEN)
        #hp_map[list(pixels)] = 0
        #hp.mollview(hp_map, title=catalog_name, coord='C', cbar=None)


        for i, checks in enumerate(self.flags_to_check):
            #quantities_this = checks['quantities']
            kind = checks['kind']
            flag_val = checks['flag_val']
            quantities_this = flags_tot[i]

            #        for quantities_this in flags_tot:
            fig = None; ax=None;
            if rank==0:
                fig, ax = plt.subplots()

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

                
                if rank==0:
                    result_this_quantity = {}
                    galaxy_count = len(recvbuf)

                    frac = np.sum(recvbuf)/(len(recvbuf)+0.0)*100.

                    xbins = np.linspace(np.min(recvbuf_ra),np.max(recvbuf_ra),50)
                    ybins = np.linspace(np.min(recvbuf_dec),np.max(recvbuf_dec),50)
                    area = (xbins[1]-xbins[0])*(ybins[1]-ybins[0])*(60.**2) #arcminutes
                    im = ax.hist2d(recvbuf_ra[recvbuf],recvbuf_dec[recvbuf], bins=(xbins,ybins),weights = 1./area*np.ones(len(recvbuf_ra[recvbuf])))#, label=quantity)


                    result_this_quantity[0] = (
                             frac,
                             ('fail' if frac<50 else 'pass'),
                             quantity,
                    )

                    result_this_quantity[1] = (
                             np.mean(im[0][im[0]>0]),
                             ('fail' if np.mean(im[0][im[0]>0]<1.0) else 'pass'),
                             quantity,
                    )
                    quantity_hashes[tuple(result_this_quantity[s][0] for s in [0,1])].add(quantity)

                    self.record_result(
                        result_this_quantity,
                        quantity ,
                        plots_tot[i]
                    )
                    
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
