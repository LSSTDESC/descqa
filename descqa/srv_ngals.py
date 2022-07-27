from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle, chain
from collections import defaultdict
import numpy as np
from mpi4py import MPI

from .base import BaseValidationTest, TestResult
from .plotting import plt
from .parallel import send_to_master, get_ra_dec

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


__all__ = ['CheckNgals']


class CheckNgals(BaseValidationTest):
    """
    Check of number of galaxies given flags and filters
    """

    def __init__(self, **kwargs):
        '''
        Read inputs from yaml file and initialize class
        '''

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


        if not any((
                self.flags_to_check,
                self.catalog_filters,
        )):
            raise ValueError('must specify flags_to_check, catalog_filters')


        self.prop_cycle = None
        self.current_catalog_name = None
        self.current_failed_count = None
        self._individual_header = list()
        self._individual_table = list()

        super(CheckNgals, self).__init__(**kwargs)


    def record_result(self, results, quantity_name=None, more_info=None, failed=None, individual_only=False):
        '''
        Record result by updating failed count and summary
        '''
        if isinstance(results, dict):
            self.current_failed_count += sum(1 for v in results.values() if v[1] == 'fail')
        elif failed:
            self.current_failed_count += 1

        if quantity_name is None:
            # update header if no quantity name specified 
            self._individual_header.append(self.format_result_header(results, failed))
        else:
            # else add a row to the table
            self._individual_table.append(self.format_result_row(results, quantity_name, more_info))



    def format_result_row(self, results, quantity_name, more_info):
        '''
        Add result, fail class and variable title to output rows
        '''
        more_info = 'title="{}"'.format(more_info) if more_info else ''
        output = ['<tr>', '<td {1}>{0}</td>'.format(quantity_name, more_info)]
        for s in range(2):
            output.append('<td class="{1}" title="{2}">{0:.4g}</td>'.format(*results[s]))
        output.append('</tr>')
        return ''.join(output)

    @staticmethod
    def format_result_header(results, failed=False):
        '''
        Add non-table result to summary 
        '''
        return '<span {1}>{0}</span>'.format(results, 'class="fail"' if failed else '')


    def generate_summary(self, output_dir):
        '''
        Generate summary table
        '''

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

        self._individual_header.clear()
        self._individual_table.clear()



    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        all_quantities = sorted(map(str, catalog_instance.list_all_quantities(True)))

        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))
        self.current_catalog_name = catalog_name
        self.current_failed_count = 0
        galaxy_count = None

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
            filters.append(filt['filters'])
        filters = list(chain(*filters))


        flags_tot =[]
        label_tot=[]
        plots_tot=[]
        for i, checks in enumerate(self.flags_to_check):
            # total list of quantities  - note do we need the quantity_pattern matching here or can we assume we know the input quantities?
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


        quantities_this_new = list(chain(*flags_tot))
        quantities_this_new.append(self.ra)
        quantities_this_new.append(self.dec)



        if len(filters) > 0:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,filters=filters,return_iterator=False, rank=rank, size=size)
        else:
            catalog_data = catalog_instance.get_quantities(quantities_this_new,return_iterator=False, rank=rank, size=size)

        recvbuf_ra, recvbuf_dec = get_ra_dec(self.ra,self.dec,catalog_data)

        if rank==0:
            galaxy_count = len(recvbuf_ra)
            self.record_result('Found {} entries in this catalog.'.format(galaxy_count))


        for i, checks in enumerate(self.flags_to_check):
            #quantities_this = checks['quantities']
            kind = checks['kind']
            flag_val = checks['flag_val']
            quantities_this = flags_tot[i]

            fig = None; ax=None;
            if rank==0:
                fig, ax = plt.subplots()

            for quantity in quantities_this:
                #PL : only currently works for doubles and booleans
                value = catalog_data[quantity] 
                recvbuf = send_to_master(value, kind)

                if rank==0:
                    flag_val=False
                    result_this_quantity = {}
                    galaxy_count = len(recvbuf)
                    recvbuf = np.logical_not(recvbuf)

                    frac = np.sum(recvbuf)/(len(recvbuf)+0.0)*100.

                    xbins = np.linspace(np.min(recvbuf_ra),np.max(recvbuf_ra),50)
                    ybins = np.linspace(np.min(recvbuf_dec),np.max(recvbuf_dec),50)
                    area = (xbins[1]-xbins[0])*(60.**2)*(np.sin(ybins[1]*np.pi/180.)-np.sin(ybins[0]*np.pi/180.))*180./np.pi 
                    # area in square arcminutes


                    im = ax.hist2d(recvbuf_ra[recvbuf],recvbuf_dec[recvbuf], bins=(xbins,ybins),weights = 1./area*np.ones(len(recvbuf_ra[recvbuf])))

                    result_this_quantity[0] = (
                             np.mean(im[0][im[0]>0]),
                             'pass',
                             quantity,
                    )
                    result_this_quantity[1] = (
                             frac,
                             'pass',
                             quantity,
                    )

                    self.record_result(
                        result_this_quantity,
                        quantity ,
                    )


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
        self.generate_summary(output_dir)
