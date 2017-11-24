from __future__ import division, unicode_literals, absolute_import
import os

__all__ = ['BaseValidationTest', 'TestResult']

class TestResult(object):
    """
    class for passing back test result
    """
    def __init__(self, score=None, summary='', passed=False, skipped=False):
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
        """

        self.skipped = bool(skipped)
        self.passed = bool(passed)
        self.summary = summary or ''

        # set score
        if not self.skipped:
            try:
                self.score = float(score)
            except (TypeError, ValueError):
                raise ValueError('Must set a float value for `score`')


class BaseValidationTest(object):
    """
    very abstract class for validation test class
    """

    data_dir = os.path.join(os.path.dirname(__file__), 'data')

    def __init__(self, **kwargs):
        pass


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        """
        Run the validation test on a single catalog.
        Return an instance of TestResult.
        This method will be called once for each catalog.

        Parameters
        ----------
        catalog_instance : instance of BaseGenericCatalog
            instance of the galaxy catalog

        catalog_name : str
            name of the galaxy catalog

        output_dir : str
            output directory (all output must be under this directory)

        Returns
        -------
        test_result : instance of TestResult
            use the TestResult object to return test result
        """
        raise NotImplementedError


    def conclude_test(self, output_dir):
        """
        Conclude the test.
        One can make summary plots for all catalogs here.
        Return None.
        This method will be called once when all catalogs are done.

        Parameters
        ----------
        output_dir: str
            output directory (all output must be under this directory)
        """
        pass
