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
from difflib import SequenceMatcher
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
        'size{}':{
            'suffixes': ['', '_disk', '_bulge', '_minor', '_minor_disk', '_minor_bulge'],
            'range_min': 0.,
            'range_max': 150.,
            'mean_min': 0.1,
            'mean_max': 10.,
        },
    }       
    #checks to run
    possible_tests = {'range':['range_min', 'range_max'],
                      'mean': ['mean_min', 'mean_max'],
                      'std': ['std_min', 'std_max'],
                     } 
    NA = 'N/A'

    #other defaults
    Nbins_default = 25
    default_markers = ['o', 'v', 's', 'd', 'H', '^', 'D', 'h', '<', '>', '.']
    default_linestyles = ['-', '--', '-.', ':']
    default_dashstyles = [[2, 2, 2, 2, 4, 2], [4, 2, 2, 2, 4, 2], [8, 4, 2, 4, 2, 4, 2, 4], [8, 4, 8, 4, 8, 4, 2, 4]] #dot-dot-dash, dash-dash-dot, ..
    possible_quantity_modifiers = ['','_true']
    yaxis_xoffset = 0.02
    yaxis_yoffset = 0.5
    figx = 4.5
    figy = 4.5

    def __init__(self, quantities=[], N_bins=[], ncolumns=2, sharex='none', **kwargs):
        """
        Perform checks on supplied list of quantities;
        quantities can be list of quantity_names (to be matched), keywords, sub-lists of quantity names, or combinations thereof
        all curves for a keyword or sub-list of quantity names will appear on the same sub-plot
        binning can be supplied or default; bins specified by '' revert to default
        """
        #catalog quantities requested
        if not quantities:
            raise ValueError('No quantities given to check')
        #check for legal quantities (must be strings or lists of strings)
        quantities_flat = [''.join(q) for q in quantities if type(q) is str] + [''.join(c) for q in quantities for c in q if type(q) is list]
        if all(type(q) is str for q in quantities_flat):
            self.quantities = quantities
        else:
            raise ValueError('Quantities to be checked must be strings or list of strings')

        #setup default or requested binning
        if N_bins:
            if len(N_bins) != len(quantities):
                raise ValueError('Mismatch in number of quantities or quantity groups ({}) and number of user-selected bins ({})', len(quantities), len(N_bins))
            else:
                #use supplied binning and default if '' supplied
                self.N_bins = [int(n) if len(str(n)) > 0 else self.Nbins_default for n in N_bins]
        else:
            self.N_bins = None

        #plotting variables
        self.ncolumns = int(ncolumns)
        self.markers = iter(self.default_markers)
        self._color_iterator = ('C{}'.format(i) for i in count())
        self.yaxis = '$N$'
        self.sharex = sharex

        #setup subplot configuration 
        self.nplots = len(self.quantities)
        self.nrows = (self.nplots+self.ncolumns-1)//self.ncolumns
        #scale fig size to number of plots requested
        self.figx_p = self.ncolumns*self.figx
        self.figy_p = self.nrows*self.figy

        #setup summary plot
        self.summary_fig, self.summary_ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p), sharex=self.sharex)
        self.summary_fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical') #setup a common axis label

        self._other_kwargs = kwargs

    def get_desired_quantities(self, q, catalog_instance):
        #check if str is a keyword
        if q in self.checks.keys():
            desired_quantities = sorted([q.format(s) if 'suffixes' in self.checks[q].keys() else q for s in self.checks[q]['suffixes']])
        else: #match string from all available quantities                                                                                  
            desired_quantities = sorted([cq for cq in catalog_instance.list_all_quantities() if q in cq])
            
        return desired_quantities

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        #update color and marker to preserve catalog colors and markers across tests
        catalog_color = next(self._color_iterator)
        version = catalog_instance.get_catalog_info('version')

        #get desired quantities
        desired_quantities = []
        for qgroup in self.quantities:
            if type(qgroup) is str:
                desired_quantities.append(self.get_desired_quantities(qgroup, catalog_instance))
            elif type(qgroup) is list:
                desired_qgroup = []
                for q in qgroup:
                    for dq in self.get_desired_quantities(q, catalog_instance):
                        desired_qgroup.append(dq)
                desired_quantities.append(desired_qgroup)
        #print('desired:',desired_quantities)

        #setup binning for qgroups
        if not self.N_bins:
            self.N_bins = [self.Nbins_default for q in desired_quantities]

        #generate possible quantities from allowed modifiers
        possible_quantities = []
        for qgroup in desired_quantities:
            possible_qgroup = [[q+m for m in self.possible_quantity_modifiers if len(m)==0 or (len(m)>0 and not m in q)] for q in qgroup]
            possible_quantities.append(possible_qgroup)

        #check catalog data for desired quantities and adjust N_bins list accordingly
        quantities=[]
        xaxis_labels=[]
        for n, pqgroup in enumerate(possible_quantities):
            qgroup = []
            for pq in pqgroup: 
                quantity = catalog_instance.first_available(*pq)
                if not quantity:
                    print("Skipping missing possible quantities {}".format(pq))
                else:
                    qgroup.append(quantity)
            if not qgroup:  #skip group if none found
                del self.Nbins[n]
            else:
                quantities.append(qgroup)
                #find common string in qgroup elements, remove '_' and use it for xaxis label
                common_strings = [SequenceMatcher(None, qgroup[0],qg).find_longest_match(0,len(qgroup[0]),0, len(qg)) for qg in qgroup]
                sizes = [match.size for match in common_strings]
                xaxis_labels.append(re.sub('_', '', qgroup[0][common_strings[sizes.index(min(sizes))].a:common_strings[sizes.index(min(sizes))].size]))
        #print('found:',quantities)

        if not quantities:
            return TestResult(skipped=True, summary='Missing all requested quantities')

        #fill out check_values based on supplied checks and available catalog quantities
        #easiest to copy the dict for every q and ignore the qgroups
        check_values = {}
        for key in self.checks.keys():
            #add allowed suffixes to this key
            keys = [key.format(s) if 'suffixes' in self.checks[key].keys() else key for s in self.checks[key]['suffixes']]
            #add allowed modifiers to keys
            matches = [k+m for m in self.possible_quantity_modifiers if len(m)==0 or (len(m)>0 and not m in k) for k in keys]

            #find corresponding quantities and populate check_values dict
            for qgroup, N in zip(quantities, self.N_bins):
                for q in qgroup:
                    if q in matches:
                        check_values[q] = self.checks[key]
                        #use binning in checks dict if available 
                        #TODO allow to override this from options
                        if 'N_bins' not in self.checks[key].keys():
                            check_values[q]['N_bins'] = N
                    else: #no match found; set bin size from options/default
                        check_values[q] = {'N_bins': N}
                        
        #setup plots
        fig, ax = plt.subplots(self.nrows, self.ncolumns, figsize=(self.figx_p, self.figy_p), sharex=self.sharex)
        fig.text(self.yaxis_xoffset, self.yaxis_yoffset, self.yaxis, va='center', rotation='vertical') #setup a common axis label

        #initialize arrays for storing data statistics and histogram sums; number of bins may vary with q
        Ntotal_list, N_list, sumq_list, sumq2_list = [], [], [], []
        for qgroup in quantities:
            Ntotal_g, N_g, sumq_g, sumq2_g = [], [], [], []
            for q in qgroup:
                Nbins = check_values[q]['N_bins']
                Ntotal_g.append(0)
                N_g.append(np.zeros(check_values[q]['N_bins'], dtype=np.int))
                sumq_g.append(np.zeros(check_values[q]['N_bins']))
                sumq2_g.append(np.zeros(check_values[q]['N_bins']))
            Ntotal_list.append(Ntotal_g)
            N_list.append(N_g)
            sumq_list.append(sumq_g)
            sumq2_list.append(sumq2_g)
        
        #get catalog data by looping over data iterator (needed for large catalogs) and aggregate histograms
        quantities_flat = [''.join(q) for qgroup in quantities for q in qgroup]
        for catalog_data in catalog_instance.get_quantities(quantities_flat, return_iterator=True):
            for ng, qgroup  in enumerate(quantities):
                for nq, q in enumerate(qgroup):
                    Ntotal_list[ng][nq] += len(catalog_data[q])

            catalog_data = GCRQuery(*((np.isfinite, col) for col in catalog_data)).filter(catalog_data)

            bin_edges_list = []
            data_min = {}
            data_max = {}
            for qgroup, N_g, sumq_g, sumq2_g in zip(quantities, N_list, sumq_list, sumq2_list):
                bin_edges_g = []
                for q, N, sumq, sumq2 in zip(qgroup, N_g, sumq_g, sumq2_g):
                    #bin catalog_data and accumulate subplot histograms
                    hist, bin_edges = np.histogram(catalog_data[q], bins=check_values[q]['N_bins'])
                    N += hist
                    sumq += np.histogram(catalog_data[q], bins=check_values[q]['N_bins'], weights=catalog_data[q])[0]
                    sumq2 += np.histogram(catalog_data[q], bins=check_values[q]['N_bins'], weights=catalog_data[q]**2)[0]
                    bin_edges_g.append(bin_edges)
                    #get max and min values
                    if catalog_data[q].dtype.char in 'bBiulfd':
                        data_min[q] = min(np.nanmin(catalog_data[q]), data_min.get(q, np.inf))
                        data_max[q] = max(np.nanmax(catalog_data[q]), data_max.get(q, -np.inf))
                bin_edges_list.append(bin_edges_g)

        #TODO check for outliers and truncate range of histograms if required

        #loop over quantities and make plots
        results = {}
        for qgroup in quantities:
            print(qgroup)

        for ax_this, summary_ax_this, qqroup, bin_edges_g, N_g, sumq_g, sumq2_g, Ntotal_g, xlabel  in zip_longest(
                ax.flat,
                self.summary_ax.flat,
                quantities, bin_edges_list, N_list, sumq_list, sumq2_list, Ntotal_list, xaxis_labels
            ):
            print('inloop',qgroup )
            if qgroup is None:
                ax_this.set_visible(False)
                summary_ax_this.set_visible(False)
            else: #loop over quantities in each subplot
                linestyles = cycle(self.default_linestyles)
                dashstyles = cycle(self.default_dashstyles)
                for nplot, (q, bin_edges, N, sumq, sumq2, Ntotal) in enumerate(zip(qgroup, bin_edges_g, N_g, sumq_g, sumq2_g, Ntotal_g)):
                    results[q] = {'range':{}, 'mean':{}, 'std':{}}
                    meanq = sumq/N
                    sumN = N.sum()
                    #save statistics in results dict
                    results[q]['mean']['value'] = sumq.sum()/sumN
                    results[q]['std']['value'] = math.sqrt(sumq2.sum()/sumN - results[q]['mean']['value']**2)
                    results[q]['range']['value'] = (data_min[q], data_max[q]) if q in data_min.keys() and q in data_max.keys() else self.NA

                    #1st pass check example if means etc. fall in range allowed by check_values
                    for tests in self.possible_tests.keys():
                        test_results = []
                        for test in self.possible_tests[tests]:
                            if test == tests+'_min':
                                if type(results[q][tests]['value']) is not tuple:
                                    test_results.append(results[q][tests]['value'] > check_values[q][tests+'_min'] if tests+'_min' in check_values[q] else self.NA)
                                #TODOD range check
                            elif test == tests+'_max':
                                if type(results[q][tests]['value']) is not tuple:
                                    test_results.append(results[q][tests]['value'] < check_values[q][tests+'_max'] if tests+'_max' in check_values[q] else self.NA)
                                #TODO range check
                        if test_results:
                            results[q][tests]['pass'] = all([res==True for res in test_results]) if not all([res==self.NA for res in test_results]) else self.NA
                        else:
                            results[q][tests]['pass'] = self.NA

                    #add histogram to subplot using built-in or custom linestyle (up to 8 allowed)
                    ls = next(linestyles) if nplot < len(self.default_linestyles) else None
                    dashes = next(dashstyles) if nplot >= len(self.default_linestyles) and nplot < len(self.default_linestyles) + len(self.default_dashstyles) else None
                    catalog_label = '$'+re.sub('_',' ',re.sub(xlabel,'',q)).strip()+'$'
                    print(nplot, q, catalog_label)
                    self.catalog_subplot(ax_this, bin_edges[:-1], N, catalog_color, catalog_label, ls=ls, dashes=dashes)

                    #add curve for this catalog to summary plot
                    self.catalog_subplot(summary_ax_this, bin_edges[:-1], N, catalog_color, catalog_label, ls=ls, dashes=dashes)
                
                    #save results for catalog in text file
                    with open(os.path.join(output_dir, 'Check_' + catalog_name + '.txt'), 'ab') as f_handle: #open file in append mode
                        comment = self.get_comment(q, Ntotal, sumN, results)
                        self.save_quantities(q, meanq, N, f_handle, comment=comment)

                self.decorate_subplot(ax_this, label='v'+version, xlabel=xlabel)
                self.decorate_subplot(summary_ax_this, label='v'+version, xlabel=xlabel)

        #make final adjustments to plots and save figure
        self.post_process_plot(fig)
        fig.savefig(os.path.join(output_dir, '_'.join(('Check',catalog_name)) + '.png'))
        plt.close(fig)
        return TestResult(0, passed=True)

    def get_comment(self, q, Ntotal, sumN, results):

        comment = 'Summary for {}\n'\
                  '    Total # of nan or inf values = {}\n'\
                  '    Total # of galaxies = {}\n'\
                  .format(q, Ntotal - sumN, sumN)
        for key in results[q].keys():
            if type(results[q][key]['value']) is tuple and len(results[q][key]['value'])==2:
                string = '   {} = ({:12.4g},{:12.4g}); TEST RESULT: {}\n'.format(key, results[q][key]['value'][0], results[q][key]['value'][1], results[q][key]['pass'])
            else:
                string = '   {} = {:12.4g}; TEST RESULT: {}\n'.format(key, results[q][key]['value'], results[q][key]['pass'])
            comment = comment + string

        return comment
        

    @staticmethod
    def catalog_subplot(ax, lower_bin_edges, data, catalog_color, catalog_label, ls=None, dashes=None):
        if dashes is not None:
            ax.step(lower_bin_edges, data, label=catalog_label, color=catalog_color, dashes=dashes)
        elif ls is not None:
            ax.step(lower_bin_edges, data, label=catalog_label, color=catalog_color, linestyle=ls)
        else:
            raise ValueError('catalog_subplot called without linestyle or dashstyle')

    @staticmethod
    def decorate_subplot(ax, xlabel=None, label=None, yscale='linear'):
        ax.tick_params(labelsize=8)
        ax.set_yticklabels(['{:.2e}'.format(float(y)) for y in ax.get_yticks().tolist()])
        ax.set_yscale(yscale)
        if label:
            ax.text(0.05, 0.97, label, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes)
        if xlabel:
            ax.set_xlabel(re.sub('_','',xlabel))
        ax.legend(loc='best', fancybox=True, framealpha=0.5, numpoints=1)

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
