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

#import lsst
#import lsst.analysis
import lsst.analysis.tools

from lsst.analysis.tools.actions.scalar import MedianAction
from lsst.analysis.tools.actions.vector import SnSelector, MagColumnNanoJansky, MagDiff
from lsst.analysis.tools.interfaces import AnalysisMetric
from lsst.analysis.tools.analysisPlots.analysisPlots import WPerpPSFPlot
from lsst.analysis.tools.tasks.base import _StandinPlotInfo

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['TestMetric','DemoMetric']


def reconfigured_wperp():
    ''''''
    # call base class
    wPerpAction = WPerpPSFPlot()

    # reconfigure
    wPerpAction.process.buildActions.x = MagDiff()
    wPerpAction.process.buildActions.x.col1 = "g_psfFlux"
    wPerpAction.process.buildActions.x.col2 = "r_psfFlux"
    wPerpAction.process.buildActions.x.returnMillimags=False
    wPerpAction.process.buildActions.y = MagDiff()
    wPerpAction.process.buildActions.y.col1 = "r_psfFlux"
    wPerpAction.process.buildActions.y.col2 = "i_psfFlux"
    wPerpAction.process.buildActions.y.returnMillimags=False
    wPerpAction.prep.selectors.snSelector.threshold = 100
 
    # populate prep
    wPerpAction.populatePrepFromProcess()

    # get list of quantities
    key_list_full = list(wPerpAction.prep.getInputSchema())
    key_list = [key_list_full[i][0] for i in range(len(key_list_full))]

    return wPerpAction, key_list




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
            wPerpAction, key_list = reconfigured_wperp()
            self.wPerpAction = wPerpAction

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
        quantities_new = comm.bcast(quantities_new, root=0)

        quantities = tuple(quantities_new)
        # note that snr is defined on flux directly and not on magnitudes

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
            if ('flag' in quantity) or ('Flag' in quantity) or ('detect' in quantity):
                recvbuf[quantity] = send_to_master(data_rank[quantity],'bool')
            else:
                recvbuf[quantity] = send_to_master(data_rank[quantity],'double')

        if rank==0:
            print(len(recvbuf[quantity]))


        if rank==0:

            #wPerpAction, key_list = reconfigured_wperp()

            stage1 = self.wPerpAction.prep(recvbuf)
            stage2 = self.wPerpAction.process(stage1)

            plot = self.wPerpAction.produce(stage2, plotInfo=_StandinPlotInfo())

            plt.savefig(output_dir+"stellarLocusTest.png")
            plt.close()

        if rank==0:
            self.generate_summary(output_dir)
        else: 
            self.current_failed_count=0
        
        self.metric=0
        self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
        self.metric = comm.bcast(self.metric, root=0)

        return TestResult(passed=(self.current_failed_count == 0), score=self.metric)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
