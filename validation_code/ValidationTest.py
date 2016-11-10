from __future__ import (division, print_function, absolute_import, unicode_literals)
from warnings import warn
import os

__all__ = ['ValidationTest', 'TestResult']

class TestResult(object):
    """
    class for passing back test result
    """
    def __init__(self, status, summary):
        """
        Parameters
        ----------
        status : str
            run status, must be "PASSED", "FAILED", or "SKIPPED"

        summary : str
            short summary, length must be less than 80
        """
        status = status.upper()
        if status not in ('PASSED', 'FAILED', 'SKIPPED'):
            raise ValueError('`status` must be "PASSED", "FAILED", or "SKIPPED"')
        if not isinstance(summary, basestring) or len(summary) > 80:
            raise ValueError('`summary` must be a string of length < 80')
        self.status = status
        self.summary = summary


class ValidationTest(object):
    """
    abstract class for validation test class
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        base_data_dir : string
            base directory that contains validation data
        
        base_output_dir : string
            base directory to store test data, e.g. plots
        """
        
        #enforce the existence of required methods
        required_method_names = ['run_validation_test']
        for required_method_name in required_method_names:
            if not hasattr(self, required_method_name):
                msg = ("Any sub-class of Validation_Test must implement a method named %s " % required_method_name)
                raise SyntaxError(msg)
        
        #check to see if directories exist
        if os.path.exists(kwargs['base_data_dir']):
            self.base_data_dir = kwargs['base_data_dir']
        else:
            msg = ('base data directory does not exist.')
            raise ValueError(msg)
        
        #enforce required keyword arguments have been passed
        required_keys = ['test_name']
        for key in required_keys:
            if key not in kwargs.keys():
                msg = ("Validation_Test must initialized with %s keyword argument" % key)
                raise ValueError(msg)


