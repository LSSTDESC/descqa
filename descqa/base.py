from __future__ import division, unicode_literals, absolute_import
import os

__all__ = ['BaseValidationTest', 'TestResult']

class TestResult(object):
    """
    class for passing back test result
    """
    def __init__(self, score=None, summary=None, passed=False, skipped=False, inspect_only=False):
        """
        Parameters
        ----------
        score : float
            a float number to represent the test score

        summary : str
            short summary string

        passed : bool
            if the test is passed

        skipped : bool
            if the test is skipped, overwrites all other arguments

        inspect_only : bool
            if the test is only for inspection (i.e., no passing criteria)
        """

        self.passed = bool(passed)
        self.skipped = bool(skipped)
        self.inspect_only = bool(inspect_only)
        self.summary = summary or ''

        if sum((self.passed, self.skipped, self.inspect_only)) > 1:
            raise ValueError('Only *one* of `passed`, `skipped`, and `inspect_only` can be set to True.')

        # set score
        if not (self.skipped or self.inspect_only):
            try:
                self.score = float(score)
            except (TypeError, ValueError):
                raise ValueError('Must set a float value for `score`')


    @property
    def status_code(self):
        """
        get status code (e.g. VALIDATION_TEST_PASSED)
        """
        if self.passed:
            return 'VALIDATION_TEST_PASSED'
        if self.skipped:
            return 'VALIDATION_TEST_SKIPPED'
        if self.inspect_only:
            return 'VALIDATION_TEST_INSPECT'
        return 'VALIDATION_TEST_FAILED'


    @property
    def status_full(self):
        """
        get full status (3 lines of string: status code, summary, score)
        """
        output = [self.status_code, self.summary]
        if self.score:
            output.append('{:.3g}'.format(self.score))
        return '\n'.join(output)


class BaseValidationTest(object):
    """
    very abstract class for validation test class
    """

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    external_data_dir = "/global/cfs/cdirs/lsst/groups/CS/descqa/data"

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
