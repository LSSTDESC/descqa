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
import time

from .base import BaseValidationTest, TestResult
from .plotting import plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckShear','shear_from_moments']



def shear_from_moments(Ixx,Ixy,Iyy,kind='chi'):
    '''
    get shear components from second moments
    '''
    if kind=='eps':
        denom = Ixx+Iyy + 2.*np.sqrt(Ixx*Iyy - Ixy**2)
    elif kind=='chi':
        denom = Ixx + Iyy # chi 
    return (Ixx-Iyy)/denom, 2*Ixy/denom



class CheckShear(BaseValidationTest):
    """
    Check shear values
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
        self.Ixx = kwargs.get('Ixx')
        self.Ixy= kwargs.get('Ixy')
        self.Iyy= kwargs.get('Iyy')
        self.IxxPSF= kwargs.get('IxxPSF')
        self.IxyPSF= kwargs.get('IxyPSF')
        self.IyyPSF= kwargs.get('IyyPSF')

        if not any((
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

        super(CheckShear, self).__init__(**kwargs)

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

        print(filters)
        lgnd_loc_dflt ='best'


        label_tot=[]
        plots_tot=[]

        quantities=[self.ra,self.dec,self.Ixx,self.Iyy,self.Ixy,self.IxxPSF, self.IyyPSF, self.IxyPSF]
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
            count = len(data_rank[quantity])
            tot_num = comm.reduce(count)
            counts = comm.allgather(count)
            if rank==0:
                recvbuf[quantity] = np.zeros(tot_num)
            else:
                recvbuf[quantity] = None
            displs = np.array([sum(counts[:p]) for p in range(size)])
            comm.Gatherv([data_rank[quantity],MPI.DOUBLE], [recvbuf[quantity], counts, displs, MPI.DOUBLE],root=0)

        e1,e2 = shear_from_moments(recvbuf[self.Ixx],recvbuf[self.Ixy],recvbuf[self.Iyy],kind='chi')

        e1psf,e2psf = shear_from_moments(recvbuf[self.IxxPSF],recvbuf[self.IxyPSF],recvbuf[self.IyyPSF],kind='chi')

        plt.figure()        
        quantity_hashes[0].add('s1')
        self.record_result((0,'s1'),'s1','p_s1.png')
        plt.hist(e1,bins=np.linspace(-1.,1.,100))
        plt.hist(e1psf,bins=np.linspace(-1.,1.,100))
        plt.savefig(os.path.join(output_dir, 'p_s1.png'))
        plt.close()
        plt.figure()
        quantity_hashes[0].add('s2')
        self.record_result((0,'s2'),'s2','p_s2.png')
        plt.hist(e2,bins=np.linspace(-1.,1.,100))
        plt.hist(e2psf,bins=np.linspace(-1.,1.,100))
        plt.savefig(os.path.join(output_dir, 'p_s2.png'))
        plt.close()
        '''plt.figure()
        quantity_hashes[0].add('s12')
        self.record_result((0,'s12'),'s12','p_s12.png')
        plt.hist2d(e1,e2,bins=100)
        plt.savefig(os.path.join(output_dir, 'p_s12.png'))
        plt.close()'''

        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
