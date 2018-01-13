from __future__ import print_function, division, unicode_literals, absolute_import
import os
import math
try:
    from itertools import zip_longest
except ImportError:
    from itertools import izip_longest as zip_longest
from itertools import count
import numpy as np
from GCR import GCRQuery

from .base import BaseValidationTest, TestResult
from .plotting import plt
from itertools import cycle
import re

__all__ = ['CheckQuantities']


class CheckQuantities(BaseValidationTest):
    """
    Readiness test to check catalog quantities before image simulations 
    """
    #setup dict with parameters needed to vet quantities
    checks = {
        'ellipticity{}': {
            'suffixes': ['', '_disk', '_bulge'],
            'N_bins': 25,
            'range_min': 0.,
            'range_max': 1.,
            'mean_min': 0.2,
            'mean_max': 0.8,
        },
        'ellipticity_{}': {
            'suffixes': ['1', '2', '1_disk', '1_bulge', '2_disk', '2_bulge'],
            'N_bins': 40,
            'range_min': -1.,
            'range_max': 1.,
            'mean_min': -0.5,
            'mean_max': 0.5,
        },
    }       
    #other defaults
    Nbins_default = 25
    default_markers = ['o', 'v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']
    possible_quantity_modifiers = ['','_true']
    yaxis_xoffset = 0.02
    yaxis_yoffset = 0.5

    def __init__(self, quantities=None, N_bins=None, ncolumns=2, **kwargs):
        #catalog quantities requested
        if quantities is None:
            raise ValueError('No quantities given to check')
        if type(quantities) is str or type(quantities) is list:
            self.quantities = quantities
        else:
            raise ValueError('Unknown format for quantities to check')

        #setup default or requested binning
        if type(N_bins) is list and type(quantities) is list:
            if len(N_bins) != len(quantities):
                raise ValueError('Mismatch in number of quantities ({}) and number of user-selected bins ({})', len(quantities), len(N_bins))
            else:
                self.N_bins = [int(n) if len(str(n)) > 0 else self.Nbins_default for n in N_bins]
        else:
            self.N_bins = None

        #plotting variables
        self.ncolumns = int(ncolumns)
        self.markers = iter(self.default_markers)
        self._color_iterator = ('C{}'.format(i) for i in count())
        self.yaxis = '$N$'

        #setup subplot configuration 
        self.nplots = len(self.quantities)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns)
        self.summary_fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical') #setup a common axis label

        self._other_kwargs = kwargs


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)

        #get desired quantities
        if type(self.quantities) is str:
            self.desired_quantities = sorted([q for q in catalog_instance.list_all_quantities() if self.quantities in q])
        elif type(quantities) is list:
            self.desired_quantities = sorted(self.quantities)

        if not self.N_bins:
            self.N_bins = [self.Nbins_default for q in self.desired_quantities]

        #generate possible quantities from allowed modifiers
        self.possible_quantities = [[q+m for m in self.possible_quantity_modifiers if len(m)==0 or (len(m)>0 and not m in q)] for q in self.desired_quantities]

        #check catalog data for desired quantities and adjust N_bins list accordingly
        quantities=[]
        for n, pq in enumerate(self.possible_quantities):
            quantity = catalog_instance.first_available(*pq)
            if not quantity:
                print("Skipping missing possible quantities {}",pq)
                del self.N_bins[n]
            else:
                quantities.append(quantity)

        if not quantities:
            return TestResult(skipped=True, summary='Missing all requested quantities')

        #fill out check_values based on supplied checks and available catalog quantities
        check_values = {}
        for key in self.checks.keys():
            #add allowed suffixes to this key
            if 'suffixes' in self.checks[key]:
                keys = [key.format(s) for s in self.checks[key]['suffixes']]
            else:
                keys = [key]
            #add allowed modifiers to keys
            matches = [k+m for m in self.possible_quantity_modifiers if len(m)==0 or (len(m)>0 and not m in k) for k in keys]

            #find corresponding quantities and populate check_values dict
            for q, N in zip(quantities, self.N_bins):
                if q in matches:
                    check_values[q] = self.checks[key]
                    #use binning in checks dict if available for now
                    #TODO allow to override this from options
                    if 'N_bins' not in self.checks[key].keys():
                        check_values[q]['N_bins'] = N

        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns)
        fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical') #setup a common axis label

        #initialize arrays for storing data totals
        Ntotals = np.zeros(len(quantities), dtype=np.int)

        #initialize arrays for storing histogram sums; number of bins may vary with q
        N_list, sumq_list, sumq2_list = [], [], []
        for q in quantities:
            Nbins = check_values[q]['N_bins']
            N_list.append(np.zeros(check_values[q]['N_bins'], dtype=np.int))
            sumq_list.append(np.zeros(check_values[q]['N_bins']))
            sumq2_list.append(np.zeros(check_values[q]['N_bins']))
                              
        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        for catalog_data in catalog_instance.get_quantities(quantities, return_iterator=True):
            #accumulate numbers of all data
            for n, q  in enumerate(quantities):
                Ntotals[n] += len(catalog_data[q])

            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)

            bin_edges_list = []
            for q, N, sumq, sumq2 in zip_longest(quantities, N_list, sumq_list, sumq2_list):
                #bin catalog_data and accumulate subplot histograms
                hist, bin_edges = np.histogram(catalog_data[q], bins=check_values[q]['N_bins'])
                N += hist
                sumq += np.histogram(catalog_data[q], bins=check_values[q]['N_bins'], weights=catalog_data[q])[0]
                sumq2 += np.histogram(catalog_data[q], bins=check_values[q]['N_bins'], weights=catalog_data[q]**2)[0]
                bin_edges_list.append(bin_edges)

        #TODO check for outliers and truncate range of histograms if required


        #loop over quantities and make plots
        results = {}
        for ax_this, summary_ax_this, q, bin_edges, N, sumq, sumq2, Ntotal in zip_longest(
                ax.flat,
                self.summary_ax.flat,
                quantities, bin_edges_list, N_list, sumq_list, sumq2_list, Ntotals,
        ):
            if q is None:
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)
            else:
                meanq = sumq/N
                sumN = N.sum()
                mean = sumq.sum()/sumN
                std = math.sqrt(sumq2.sum()/sumN - mean**2)
                
                #TODO check if means etc. fall in range allowed by check_values

                #make subplot
                catalog_label = ' '.join((catalog_name, '$'+q+'$'))
                #results[q] = {'bin-means': meanq, 'N':N, 'total':sumN, 'mean': mean, 'std':std}
                self.catalog_subplot(ax_this, bin_edges[:-1], N, q, catalog_color, catalog_label)

                #add curve for this catalog to summary plot
                self.catalog_subplot(summary_ax_this, bin_edges[:-1], N, q, catalog_color, catalog_label)
                
                #save results for catalog in text file
                with open(os.path.join(output_dir, 'Check_' + catalog_name + '.txt'), 'ab') as f_handle: #open file in append mode
                    comment = 'Summary for {}\n Total # of nan or inf values = {}\n Total # of galaxies = {}\n Mean = {:12.4g}; Std. Devn = {:12.4g}\n'.format(q,Ntotal - sumN, sumN, mean, std)
                    self.save_quantities(q, meanq, N, f_handle, comment=comment)

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, '_'.join(('Check',catalog_name)) + '.png'))
        plt.close(fig)
        return TestResult(0, passed=True)


    def catalog_subplot(self, ax, lower_bin_edges, data, xlabel, catalog_color, catalog_label):

        ax.step(lower_bin_edges, data, label=catalog_label, color=catalog_color)
        ax.set_yscale('log')
        ax.set_xlabel(re.sub('_','',xlabel))

    @staticmethod
    def post_process_plot(fig):
        fig.tight_layout()

    @staticmethod
    def save_quantities(q, meanq, N, filename, comment=''):
        header = 'Data columns are: <{}>, N'.format(q)
        np.savetxt(filename, np.vstack((meanq, N)).T, fmt='%12.4e', header=comment+header)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_fig)
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
