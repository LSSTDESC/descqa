from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['CheckColors']

class CheckColors(BaseValidationTest):
    """
    An example validation test
    """
    def __init__(self, **kwargs):

        # load test config options
        self.kwargs = kwargs
        ##self.option1 = kwargs.get('option1', 'option1_default')
        ##self.option2 = kwargs.get('option2', 'option2_default')
        self.test_name = kwargs.get('test_name', 'CheckColors')

        # load validation data
        ##with open(os.path.join(self.data_dir, 'README.md')) as f:
        ##    self.validation_data = f.readline().strip()

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()


    def post_process_plot(self, ax):
        #ax.text(0.05, 0.95, self.validation_data)
        ax.text(-0.25, 1.80, 'g-r vs r-i')
        #ax.legend()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # check if needed quantities exist
        if not catalog_instance.has_quantities(['mag_g_lsst', 'mag_r_lsst','mag_i_lsst']):
            return TestResult(skipped=True, summary='do not have needed quantities')

        #data = np.random.rand(10) #do your calculation with catalog_instance
        data = catalog_instance.get_quantities(['redshift','mag_u_lsst','mag_g_lsst','mag_r_lsst','mag_i_lsst','mag_z_lsst'])
        grcolor = data['mag_g_lsst'] - data['mag_r_lsst']
        ricolor = data['mag_r_lsst'] - data['mag_i_lsst']

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            #ax_this.plot(data, label=catalog_name)
            ax_this.set_xlabel('r-i',fontsize=20)		
            ax_this.set_ylabel('g-r',fontsize=20)
            ax_this.hexbin(ricolor,grcolor,gridsize=(100),cmap='GnBu',mincnt=1,bins='log')

        self.post_process_plot(ax)
        fig.savefig(os.path.join(output_dir, 'colors.png'))
        plt.close(fig)

        score = data[0] #calculate your summary statistics
        return TestResult(score, passed=True)


    def conclude_test(self, output_dir):
        self.post_process_plot(self.summary_ax)
        self.summary_fig.savefig(os.path.join(output_dir, 'colors.png'))
        plt.close(self.summary_fig)
