from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['ExampleTest']

class ExampleTest(BaseValidationTest):
    """
    An example validation test
    """
    def __init__(self, **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.option1 = kwargs.get('option1', 'option1_default')
        self.option2 = kwargs.get('option2', 'option2_default')
        self.test_name = kwargs.get('test_name', 'example_test')

        # load validation data
        with open(os.path.join(self.data_dir, 'README.md')) as f:
            self.validation_data = f.readline().strip()

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # check if needed quantities exist
        if not catalog_instance.has_quantities(['ra', 'dec']):
            return TestResult(skipped=True, summary='do not have needed quantities')

        data = np.random.rand(10) #do your calculation with catalog_instance

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            ax_this.plot(data, label=catalog_name)
            ax_this.text(0.05, 0.95, self.validation_data)

        ax.legend()
        fig.savefig(os.path.join(output_dir, 'plot.png'))
        plt.close(fig)

        score = data[0] #calculate your summary statistics
        return TestResult(score, passed=True)


    def conclude_test(self, output_dir):
        self.summary_ax.legend()
        self.summary_fig.savefig(os.path.join(output_dir, 'summary.png'))
        plt.close(self.summary_fig)
