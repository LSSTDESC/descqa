from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['CheckColors']

class CheckColors(BaseValidationTest):
    """
    Inspection test to represent 2D color plots
    """
    def __init__(self, xcolor='ri', ycolor='gr', magtype = 'mag_lsst', **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'CheckColors')
        self.possible_mag_fields = ('mag_{}_lsst',
                                    'mag_{}_sdss',
                                    'mag_{}_des',
                                    'mag_true_{}_lsst',
                                    'mag_true_{}_sdss',
                                    'mag_true_{}_des',
                                   )
        self.magtype = magtype
        # prepare summary plot
        if len(xcolor)!=2 or len(ycolor)!=2:
            print('Warning: color string is longer than 2 characters. Only first and last bands will be used.')	 
        if magtype == 'mag_lsst':
            self.xcolor1 = 'mag_{}_lsst'.format(xcolor[0])
            self.xcolor2 = 'mag_{}_lsst'.format(xcolor[-1])
            self.ycolor1 = 'mag_{}_lsst'.format(ycolor[0])
            self.ycolor2 = 'mag_{}_lsst'.format(ycolor[-1])
        else:
            raise ValueError('Magnitude type {} not implemented'.format(magtype))
            sys.exit()

        self.summary_fig, self.summary_ax = plt.subplots()


    def post_process_plot(self, ax):
        #ax.text(-0.25, 1.80, '{}-{} vs {}-{}'.format(self.ycolor1,self.ycolor2,self.xcolor1,self.xcolor2))
        ax.set_title('Color inspection for {}'.format(self.magtype))

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # check if needed quantities exist
        if not catalog_instance.has_quantities([self.xcolor1,self.xcolor2,self.ycolor1,self.ycolor2]):
            
            return TestResult(skipped=True, summary='Missing magnitudes in catalog')

        data = catalog_instance.get_quantities([self.xcolor1,self.xcolor2,self.ycolor1,self.ycolor2])
        xcolor = data[self.xcolor1] - data[self.xcolor2]
        ycolor = data[self.ycolor1] - data[self.ycolor2]

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            #ax_this.plot(data, label=catalog_name)
            ax_this.set_xlabel('{} - {}'.format(self.xcolor1,self.xcolor2),fontsize=14,labelpad=2)		
            ax_this.set_ylabel('{} - {}'.format(self.ycolor1,self.ycolor2),fontsize=14,labelpad=2)
            ax_this.hexbin(xcolor,ycolor,gridsize=(100),cmap='GnBu',mincnt=1,bins='log')

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'colors.png'))
        plt.close(fig)

        #score = data[0] #calculate your summary statistics
        return TestResult(inspect_only=True)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'colors.png'))
        plt.close(self.summary_fig)
