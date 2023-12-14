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
import sys

from .base import BaseValidationTest, TestResult
from .plotting import plt

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


__all__ = ['CheckShear','shear_from_moments']



def shear_from_moments(Ixx,Ixy,Iyy,kind='eps'):
    '''
    Get shear components from second moments
    '''
    if kind=='eps':
        denom = Ixx + Iyy + 2.*np.sqrt(Ixx*Iyy - Ixy**2)
    elif kind=='chi':
        denom = Ixx + Iyy 
    return (Ixx-Iyy)/denom, 2*Ixy/denom

def size_from_moments(Ixx, Iyy):
    return Ixx + Iyy

class CheckEllipticity(BaseValidationTest):
    """
    Check ellipticity values and second moments 
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
        self.psf_fwhm = kwargs.get('psf_fwhm')
        self.bands = kwargs.get('bands')

        if not any((
                self.catalog_filters,
        )):
            raise ValueError('you need to specify catalog_filters for these checks, add an extendedness flag, a good flag and a magnitude range')

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

        super(CheckEllipticity, self).__init__(**kwargs)

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

    def plot_moments_band(self, Ixx,Ixy,Iyy,band,output_dir):
        '''
        Plot moments for each band
        '''
        plt.figure()
        bins = np.logspace(-1.,1.5,100)
        bins_mid = (bins[1:]+bins[:-1])/2.
        Ixx_out, bin_edges = np.histogram(Ixx, bins=bins)
        Ixy_out, bin_edges = np.histogram(Ixy, bins=bins)
        Iyy_out, bin_edges = np.histogram(Iyy, bins=bins)
        self.record_result((0,'moments_'+band),'moments_'+band,'moments_'+band+'.png')
        plt.plot(bins_mid,Ixx_out,'b',label='Ixx')
        plt.plot(bins_mid,Iyy_out,'r--',label='Iyy')
        plt.plot(bins_mid,Ixy_out,'k-.',label='Ixy')
        plt.yscale('log')
        plt.xscale('log')
        plt.title(band)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'moments_'+band+'.png'))
        plt.close()
        return

    def plot_ellipticities_band(self,e1,e2,band,output_dir):
        '''
        Plot elliptiticies for each band
        '''
        plt.figure()
        bins = np.linspace(0.,1.,51)
        bins_mid = (bins[1:]+bins[:-1])/2.
        e1_out, bin_edges = np.histogram(e1, bins=bins)
        e2_out, bin_edges = np.histogram(e2, bins=bins)
        self.record_result((0,'ell_'+band),'ell_'+band,'ell_'+band+'.png')
        plt.plot(bins_mid,e1_out,'b',label='e1')
        plt.plot(bins_mid,e2_out,'r--',label='e2')
        plt.yscale('log')
        #plt.xscale('log')
        plt.title(band)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'ell_'+band+'.png'))
        plt.close()
        return

    def plot_psf(self,fwhm,band,output_dir):
        '''
        Plot PSF for each band
        '''
        plt.figure()
        bins = np.linspace(0.,1.5,201)
        bins_mid = (bins[1:]+bins[:-1])/2.
        fwhm_out, bin_edges = np.histogram(fwhm, bins=bins)
        self.record_result((0,'psf_'+band),'psf_'+band,'psf_'+band+'.png')
        plt.plot(bins_mid,fwhm_out,'b',label='psf')
        plt.yscale('log')
        plt.xlim([0.5,1.4])
        plt.title(band)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'psf_'+band+'.png'))
        plt.close()
        return

    def plot_e1e2_residuals(self,e1,e2,epsf1,epsf2,band,output_dir):
        '''
        Plot e1,e2 residuals with respect to model
        '''
        plt.figure()
        bins = np.linspace(-1.,1.,201)
        bins_mid = (bins[1:]+bins[:-1])/2.
        e1residual_out, bin_edges = np.histogram(e1-epsf1, bins=bins)
        e2residual_out, bin_edges = np.histogram(e2-epsf2, bins=bins)
        self.record_result((0,'e1e2_residuals_'+band),'e1e2_residuals_'+band,'e1e2_residuals_'+band+'.png')
        plt.plot(bins_mid,e1residual_out,'b',label='e1-e1psf')
        plt.plot(bins_mid,e2residual_out,'r--',label='e2-e2psf')
        plt.title('e1/e2 residuals vs model, band'+band)
        plt.savefig(os.path.join(output_dir, 'e1e2_residuals_'+band+'.png'))
        plt.close()
        return
        
    def plot_Tfrac_residuals(self,T,Tpsf,band,output_dir):
        '''
        Plot T fractional residuals with respect to model
        '''
        plt.figure()
        bins = np.linspace(-0.1,0.1,201)
        bins_mid = (bins[1:]+bins[:-1])/2.
        tresidual_out, bin_edges = np.histogram((T-Tpsf)/Tpsf, bins=bins)
        self.record_result((0,'tfrac_residuals_'+band),'tfrac_residuals_'+band,'tfrac_residuals_'+band+'.png')
        plt.plot(bins_mid,tresidual_out,'b',label='(T-Tpsf)/Tpsf')
        plt.title('T fractional residuals vs model, band'+band)
        plt.savefig(os.path.join(output_dir, 'tfrac_residuals_'+band+'.png'))
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

        #quantities=[self.ra,self.dec,self.Ixx,self.Iyy,self.Ixy,self.IxxPSF, self.IyyPSF, self.IxyPSF]

        # doing everything per-band first of all
        for band in self.bands:
            quantities=[self.Ixx+'_'+band,self.Iyy+'_'+band,self.Ixy+'_'+band,self.IxxPSF+'_'+band, self.IyyPSF+'_'+band, self.IxyPSF+'_'+band, self.psf_fwhm+'_'+band]
            quantities = tuple(quantities)



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
                if has_mpi:
                    if rank==0:
                        kind = get_kind(data_rank[quantity][0]) # assumes at least one element of data on rank 0
                    else:
                        kind = ''
                    kind = comm.bcast(kind, root=0)
                    recvbuf[quantity] = send_to_master(data_rank[quantity],kind)
                else:
                    recvbuf[quantity] = data_rank[quantity]


            e1,e2 = shear_from_moments(recvbuf[self.Ixx+'_'+band],recvbuf[self.Ixy+'_'+band],recvbuf[self.Iyy+'_'+band])
            e1psf,e2psf = shear_from_moments(recvbuf[self.IxxPSF+'_'+band],recvbuf[self.IxyPSF+'_'+band],recvbuf[self.IyyPSF+'_'+band])
            T = size_from_moments(recvbuf[self.Ixx+'_'+band],recvbuf[self.Iyy+'_'+band])
            Tpsf = size_from_moments(recvbuf[self.Ixx+'_'+band],recvbuf[self.Iyy+'_'+band])
             
            Ixx = recvbuf[self.Ixx+'_'+band]
            Iyy = recvbuf[self.Iyy+'_'+band]
            Ixy = recvbuf[self.Ixy+'_'+band]
            fwhm = recvbuf[self.psf_fwhm+'_'+band]

            self.plot_moments_band(Ixx,Ixy,Iyy,band,output_dir)
            self.plot_ellipticities_band(e1,e2,band,output_dir)
            self.plot_psf(fwhm,band,output_dir)
            self.plot_e1e2_residuals(e1,e2,e1psf,e2psf,band,output_dir)
            self.plot_Tfrac_residuals(T,Tpsf,band,output_dir)


        # plot moments directly per filter. For good, star, galaxy
        # FWHM of the psf
        # calculate ellpiticities and make sure they're alright 
        # look at different bands
        # note that we want to look by magnitude or SNR to understand the longer tail in moments
        # PSF ellipticity whisker plot?
        # look at what validate_drp is

        # s1/s2 plots

        # look at full ellipticity distribution test as well
        #https://github.com/LSSTDESC/descqa/blob/master/descqa/EllipticityDistribution.py
        #DC2 validation github - PSF ellipticity
        # https://github.com/LSSTDESC/DC2-analysis/blob/master/validation/Run_1.2p_PSF_tests.ipynb

        #https://github.com/LSSTDESC/DC2-analysis/blob/master/validation/DC2_calexp_src_validation_1p2.ipynb

        # Look at notes here:
        #https://github.com/LSSTDESC/DC2-production/issues/340

        # get PSF FWHM directly from data, note comments on here:
        # https://github.com/LSSTDESC/DC2-analysis/blob/u/wmwv/DR6_dask_refactor/validation/validate_dc2_run2.2i_object_table_dask.ipynb about focussing of the "telescope"


        '''mask_finite = np.isfinite(e1)&np.isfinite(e2)
        bs_out = bs(e1[mask_finite],values = e2[mask_finite],bins=100,statistic='mean')
        plt.figure()
        quantity_hashes[0].add('s1s2')
        self.record_result((0,'s1s2'),'s1s2','p_s1s2.png')
        plt.plot(bs_out[1][1:],bs_out[0])
        plt.savefig(os.path.join(output_dir, 'p_s1s2.png'))
        plt.close()


        plt.figure()        
        quantity_hashes[0].add('s1')
        self.record_result((0,'s1'),'s1','p_s1.png')
        #plt.hist(e1,bins=np.linspace(-1.,1.,100))
        plt.hist(e1psf,bins=100)#np.linspace(-1.,1.,100))
        plt.savefig(os.path.join(output_dir, 'p_s1.png'))
        plt.close()
        plt.figure()
        quantity_hashes[0].add('s2')
        self.record_result((0,'s2'),'s2','p_s2.png')
        #plt.hist(e2,bins=np.linspace(-1.,1.,100))
        plt.hist(e2psf,bins=100)#np.linspace(-1.,1.,100))
        plt.savefig(os.path.join(output_dir, 'p_s2.png'))
        plt.close()'''
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
        
        if has_mpi:
            self.current_failed_count = comm.bcast(self.current_failed_count, root=0)
           
        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
