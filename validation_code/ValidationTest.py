from __future__ import (
    division, print_function, absolute_import, unicode_literals)
from warnings import warn

class ValidationTest(object):
    """
    abstract class for validation test class
    """
    
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs :
        """
        
        # Enforce the requirement that sub-classes have been configured properly
        required_method_names = ['load_galaxy_catalog','plot_results','write_result_file',\
                                 'write_comparison_file','load_validation_data']
        
        for required_method_name in required_method_names:
            if not hasattr(self, required_method_name):
                raise SyntaxError("Any sub-class of Validation_Test must "
                    "implement a method named %s " % required_method_name)
        
