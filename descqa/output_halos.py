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

from .base import BaseValidationTest, TestResult
from .plotting import plt

import h5py

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckQuantities']

def split_for_natural_sort(s):
    """
    split a string *s* for natural sort.
    """
    return tuple((int(y) if y.isdigit() else y for y in re.split(r'(\d+)', s)))


class CheckQuantities(BaseValidationTest):
    """
    Test to read-in and output halo quantities 
    """


    def __init__(self, **kwargs):
        self.quantities_to_check = kwargs.get('quantities_to_output', [])
        self.catalog_filters = kwargs.get('catalog_filters', [])

        self.lgndtitle_fontsize = kwargs.get('lgndtitle_fontsize', 12)
        self.truncate_cat_name = kwargs.get('truncate_cat_name', False)
        self.no_version = kwargs.get('no_version', False)
        self.title_size = kwargs.get('title_size', 'small')
        self.font_size	= kwargs.get('font_size', 12)
        self.legend_size = kwargs.get('legend_size', 'x-small')
        
        if not any((
                self.quantities_to_check,
        )):
            raise ValueError('must specify quantities_to_output')

        if not all(d.get('quantities') for d in self.quantities_to_check):
            raise ValueError('yaml file error: `quantities` must exist for each item in `quantities_to_output`')

        '''if self.catalog_filters:
            if not all(d.get('filter') for d in self.catalog_filters):
                raise ValueError('yaml file error: `quantity` must exist for each item in `catalog_filters`')

            if not all(d.get('min') for d in self.catalog_filters) or all(d.get('max') for d in self.catalog_filters):
                raise ValueError('yaml file error: `min` or `max` must exist for each item in `catalog_filters`')
        '''

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

        super(CheckQuantities, self).__init__(**kwargs)

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
        for s in self.stats:
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
            self.record_result('Running halo input test on {} {}'.format(
                catalog_name,
                getattr(catalog_instance, 'version', ''),
                individual_only=True,
            ))

        if self.truncate_cat_name:
            catalog_name = catalog_name.partition("_")[0]
        version = getattr(catalog_instance, 'version', '') if not self.no_version else ''

        # check filters

        '''
        filters = []
        filter_labels = ''
        for d in self.catalog_filters:
            if rank==0:
                self.print(d)
            fq = d.get('quantity')
            if fq in all_quantities:
                filter_label=''
                qlabel = d.get('label') if d.get('label') else fq
                if d.get('min') is not None:
                    filters.append('{} >= {}'.format(fq, d.get('min')))
                    filter_label = '{} <= {}'.format(d.get('min'), qlabel)
                if d.get('max') is not None:
                    filters.append('{} < {}'.format(fq, d.get('max')))
                    flabel = '{} <= {}'.format(qlabel, d.get('max'))
                    filter_label = flabel if len(filter_label)==0 else re.sub(str(d.get('label')), flabel, filter_label)
                filter_labels = '$'+filter_label+'$' if len(filter_labels)==0 else ', '.join([filter_labels,
                                                                                              '$'+filter_label+'$'])
            else:
                self.record_result('Found no matching quantity for filtering on {}'.format(fq), failed=True)
                continue

        print(filters, filter_labels)
        '''
        lgnd_loc_dflt ='best'
        import time 
        a = time.time()
        quantity_tot =[]

        for i, checks in enumerate(self.quantities_to_check):
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

            quantities_this = sorted(quantities_this, key=split_for_natural_sort)
            quantity_tot.append(quantities_this)

        for i, filt in enumerate(self.catalog_filters):
            filters = filt['filters']


        #filters = self.catalog_filters
        #print(filters)

    
        quantities_this_new=[]
        for q in quantity_tot:
            print(q)
            if len(q)>1:
                for j in q:
                    quantities_this_new.append(j)
            else:
                quantities_this_new.append(q[0])
        quantities_this_new = tuple(quantities_this_new)
        print(quantities_this_new)

        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,filters=filters,return_iterator=False, rank=rank, size=size)
        else:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,return_iterator=False, rank=rank, size=size)
        b = time.time()
        print('read-in time = ',b-a)

        if rank==0:
            f = h5py.File('/project/projectdirs/lsst/projecta/lsst/groups/CS/halo_tmp/halo_moritz_full.hdf5','w')

        for quantity in quantities_this_new:
                #PL : only currently works for doubles 
                value = catalog_data[quantity] 
                count = len(value)
                tot_num = comm.reduce(count)
                counts = comm.allgather(count)
                if rank==0:
                    recvbuf = np.zeros(tot_num)
                else:
                    recvbuf = None
                displs = np.array([sum(counts[:p]) for p in range(size)])
                comm.Gatherv([value,MPI.DOUBLE], [recvbuf,counts,displs,MPI.DOUBLE],root=0)

                if rank==0:
                    print(time.time()-a,'read data',rank)
                    dset = f.create_dataset(quantity, data=recvbuf)

        if rank==0:
            f.close()
            print("finished writing file")


        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)