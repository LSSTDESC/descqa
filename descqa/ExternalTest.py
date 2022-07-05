from __future__ import unicode_literals, absolute_import, division
import os
import numpy as np
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['ExternalTest']

class ExternalTest(BaseValidationTest):
    """
    A class to hold external validation tests
    """
    def __init__(self, **kwargs):

        # load test config options
        self.kwargs = kwargs
        self.report_directory = kwargs.get('report_directory', './')

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        # copy plots and text files to output directory
        cmd = 'cp {}/*.txt {}/.'.format(self.report_directory,output_dir)
        os.system(cmd)
        cmd = 'cp {}/*.png {}/.'.format(self.report_directory,output_dir)
        os.system(cmd)
        return None

    def conclude_test(self, output_dir):
        return None