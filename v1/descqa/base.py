import os

__all__ = ['BaseValidationTest']

class BaseValidationTest(object):
    """
    very abstract class for validation test class
    """

    def run_validation_test(self, catalog_instance, catalog_name, output_dir):
        raise NotImplementedError

    def plot_summary(self, output_file, catalog_list):
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
        if not hasattr(self, '_catalog_list'):
            self._catalog_list = list()
        self._catalog_list.append((catalog_name, output_dir))
        return self.run_validation_test(catalog_instance, catalog_name, output_dir)

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
        catalog_list = getattr(self, '_catalog_list', list())
        catalog_list.sort(key=lambda t: t[0])
        self.plot_summary(os.path.join(output_dir, 'summary_plot.png'), catalog_list)
