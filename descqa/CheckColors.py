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
    def __init__(self, xcolor='ri', ycolor='gr', **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.test_name = kwargs.get('test_name', 'CheckColors')
        self.possible_mag_fields = ('mag_{}_lsst',
                                    'mag_{}_sdss',
                                    'mag_{}_des',
                                    'mag_{}_stripe82',
                                    'mag_true_{}_lsst',
                                    'mag_true_{}_sdss',
                                    'mag_true_{}_des',
                                    'Mag_true_{}_des_z01',
                                    'Mag_true_{}_sdss_z01'
                                    )
        # prepare summary plot
        if len(xcolor)!=2 or len(ycolor)!=2:
            print('Warning: color string is longer than 2 characters. Only first and last bands will be used.')	 
        #if magtype == 'mag_lsst':
        self.xcolor1 = xcolor[0]
        self.xcolor2 = xcolor[-1]
        self.ycolor1 = ycolor[0]
        self.ycolor2 = ycolor[-1]
        #else:
        #    raise ValueError('Magnitude type {} not implemented'.format(magtype))
        #    sys.exit()

        self.summary_fig, self.summary_ax = plt.subplots()


    def post_process_plot(self, ax, name):
        #ax.text(-0.25, 1.80, '{}-{} vs {}-{}'.format(self.ycolor1,self.ycolor2,self.xcolor1,self.xcolor2))
        ax.set_title('Color inspection for {}'.format(name))

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        for mag_field in self.possible_mag_fields:
            # check if needed quantities exist
            xcolor1_string = mag_field.format(self.xcolor1)
            xcolor2_string = mag_field.format(self.xcolor2)
            ycolor1_string = mag_field.format(self.ycolor1)
            ycolor2_string = mag_field.format(self.ycolor2)
            if not catalog_instance.has_quantities([xcolor1_string,xcolor2_string,ycolor1_string,ycolor2_string]):
                continue
            #return TestResult(skipped=True, summary='Missing magnitudes in catalog')
            data = catalog_instance.get_quantities([xcolor1_string,xcolor2_string,ycolor1_string,ycolor2_string])
            xcolor = data[xcolor1_string] - data[xcolor2_string]
            ycolor = data[ycolor1_string] - data[ycolor2_string]

            fig, ax = plt.subplots()

            for ax_this in (ax, self.summary_ax):
                ax_this.set_xlabel('{} - {}'.format(xcolor1_string,xcolor2_string),fontsize=14,labelpad=2)		
                ax_this.set_ylabel('{} - {}'.format(ycolor1_string,ycolor2_string),fontsize=14,labelpad=2)
                ax_this.hexbin(xcolor,ycolor,gridsize=(100),cmap='GnBu',mincnt=1,bins='log')

                self.post_process_plot(ax, catalog_name)
                fig.savefig(os.path.join(output_dir, '{}{}_{}{}_{}_{}.png'.format(self.xcolor1,self.xcolor2,self.ycolor1,self.ycolor2,mag_field,catalog_name)))
                plt.close(fig)

        #score = data[0] #calculate your summary statistics
        return TestResult(inspect_only=True)


    def conclude_test(self, output_dir):
        #self.post_process_plot(self.summary_ax, catalog_name)
        #self.summary_fig.savefig(os.path.join(output_dir, 'colors.png'))
        plt.close(self.summary_fig)
