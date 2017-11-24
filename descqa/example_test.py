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
            self.data = f.readline().strip()

        # prepare summary plot
        self.summary_fig, self.summary_ax = plt.subplots()


    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):

        # check if needed quantities exist
        if not galaxy_catalog.has_quantities(['ra', 'dec']):
            return TestResult(skipped=True, summary='do not have needed quantities')

        fig, ax = plt.subplots()

        for ax_this in (ax, self.summary_ax):
            ax_this.plot(np.random.rand(10), label=catalog_name)

        fig.savefig(os.path.join(base_output_dir, 'plot.png'))
        plt.close(fig)


    def generate_summary(self, catalog_name_list, base_output_dir):

        self.summary_fig.savefig(os.path.join(base_output_dir, 'summary.png'))
        plt.close(self.summary_fig)
