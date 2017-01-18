__all__ = ['ValidationTest', 'TestResult']


class TestResult(object):
    """
    class for passing back test result
    """
    def __init__(self, score=None, summary='', passed=False, skipped=False, **kwargs):
        """
        Parameters
        ----------
        score : float or None
            a float number to represent the test score

        summary : str
            short summary string

        passed : bool
            if the test is passed

        skipped : bool
            if the test is skipped, overwrites all other arguments

        **kwargs :
            any other keyword arguments
        """

        self.skipped = bool(skipped)
        self.passed = bool(passed)
        self.summary = str(summary).strip()
        for k, v in kwargs:
            self.setattr(k, v)
        
        # set score
        if not self.skipped:
            try:
                self.score = float(score)
            except (TypeError, ValueError):
                if isinstance(score, basestring) and score.upper() in ('PASSED', 'FAILED', 'SKIPPED'):
                    # this is for backward compatibility in other validations
                    status = score.upper()
                    self.passed = (status == 'PASSED')
                    self.skipped = (status == 'SKIPPED')
                else:
                    raise ValueError('Must set a float value for `score`')


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
                msg = ("Any sub-class of ValidationTest must implement a method named %s " % required_method_name)
                raise ValueError(msg)
        
        #enforce required keyword arguments have been passed
        required_keys = ['base_data_dir', 'test_name']
        for key in required_keys:
            if key not in kwargs:
                msg = ("ValidationTest must initialized with %s keyword argument" % key)
                raise ValueError(msg)
            setattr(self, key, kwargs[key])


