from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne
from scipy.stats import norm, binned_statistic
import time

from .base import BaseValidationTest, TestResult
from .plotting import plt

import lsst.analysis.tools

from lsst.analysis.tools.actions.scalar import MedianAction
from lsst.analysis.tools.actions.vector import SnSelector, MagColumnNanoJansky, MagDiff
from lsst.analysis.tools.interfaces import AnalysisMetric
from lsst.analysis.tools.analysisPlots.analysisPlots import WPerpPSFPlot, ShapeSizeFractionalDiffScatterPlot
from lsst.analysis.tools.analysisMetrics import ShapeSizeFractionalMetric
from lsst.analysis.tools.tasks.base import _StandinPlotInfo


if 'mpi4py' in sys.modules:
    from mpi4py import MPI
    from .parallel import send_to_master, get_kind
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    has_mpi = True
    print('Using parallel script, invoking parallel read')
else:
    size = 1
    rank = 0
    has_mpi = False
    print('Using serial script')


__all__ = ['TestMetric']


def reconfigured_shapesize(band):
    ''' Call analysis tools wperpPSFPlot and reconfigure for DP0.2'''
    # call base class
    shapesize = ShapeSizeFractionalMetric()
    shapesize_plot = ShapeSizeFractionalDiffScatterPlot()
    shapesize_plot.produce.addSummaryPlot = False
    # config 
    shapesize_plot.prep.selectors.snSelector.bands = ["r"]

    #shapesize_plot.prep.bands=['r']
 
    # populate prep
    shapesize.populatePrepFromProcess()
    shapesize_plot.populatePrepFromProcess()


    # get list of quantities
    key_list_full = list(shapesize.prep.getInputSchema())
    key_list = [key_list_full[i][0] for i in range(len(key_list_full))]
    key_list_full2 = list(shapesize_plot.prep.getInputSchema())
    key_list2 = [key_list_full2[i][0] for i in range(len(key_list_full2))]
    key_list.extend(key_list2)

    return shapesize, shapesize_plot, key_list




class TestMetric(BaseValidationTest):
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

        super(TestMetric, self).__init__(**kwargs)

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

        # note: add quantity list and analysis tools version to output for reproducibility 
        if rank==0:
            shapesize, shapesize_plot, key_list = reconfigured_shapesize('r')
            self.shapesize = shapesize
            self.shapesize_plot = shapesize_plot

            quantities_new = []
            for key in key_list:
                if '{band}' in key:
                    for band in self.bands:
                        quantities_new.append(key.format(band=band))
                else:
                    quantities_new.append(key)
            quantities_new = np.unique(quantities_new)
            print(quantities_new)
        else:
            quantities_new = np.array([])

        if has_mpi:
            quantities_new = comm.bcast(quantities_new, root=0)

        quantities = tuple(quantities_new)
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
            if has_mpi:
                if rank==0:
                    kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
                else:
                    kind = ''
                kind = comm.bcast(kind, root=0)
                recvbuf[quantity] = send_to_master(data_rank[quantity],kind)
            else:
                recvbuf[quantity] = data_rank[quantity]
        if rank==0:
            print(len(recvbuf[quantity]))


        if rank==0:

            # for a metric we just call it directly, and it'll output the metric
            metric = self.shapesize(recvbuf,band='r')
            for key in metric.keys():
                print(metric[key])

            # for a plot we divide it in parts so we can output the plot
            stage1 = self.shapesize_plot.prep(recvbuf,band='r')
            stage2 = self.shapesize_plot.process(recvbuf,band='r')
            plot = self.shapesize_plot.produce(stage2, plotInfo=_StandinPlotInfo(), band='r',skymap='DC2')
            plt.savefig(output_dir+"shapesizeTest.png")
            plt.close()

            '''result_this_quantity = {}
                result_this_quantity[s] = (
                    s_value,
                    'none' if flag is None else ('fail' if flag else 'pass'),
                    checks.get(s, ''),
                    )
                    print(time.time()-a,'stats checks')
                    quantity_hashes[tuple(result_this_quantity[s][0] for s in self.stats)].add(quantity)
                    self.record_result(
                        result_this_quantity,
                        quantity,
                    )'''


            for key in metric.keys():
                self.record_result({key: [metric[key], 'pass']},key)
                

        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.metric=0
        if has_mpi:
            self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
            self.metric = comm.bcast(self.metric, root=0)

        return TestResult(passed=(self.current_failed_count == 0), score=self.metric)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
