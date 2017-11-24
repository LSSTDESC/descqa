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


    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        """
        run the validation test

        Parameters
        ----------
        galaxy_catalog : galaxy catalog reader object
            instance of a galaxy catalog reader

        catalog_name : str
            name of galaxy catalog

        base_output_dir : str
            output directory

        Returns
        -------
        test_result : TestResult object
            use the TestResult object to return test result
        """
        raise NotImplementedError


    def generate_summary(self, catalog_name_list, base_output_dir):
        """
        Make summary plot (or other stuff) for all catalogs.
        Returns None.

        Parameters
        ----------
        catalog_name_list: list of str
            list of catalog names

        base_output_dir: str
            output directory
        """
        pass
