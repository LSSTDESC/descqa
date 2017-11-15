from __future__ import unicode_literals, absolute_import
import os
from builtins import str
import yaml
from .ValidationTest import BaseValidationTest, TestResult

__all__ = ['ListAvailableQuantities']

class ListAvailableQuantities(BaseValidationTest):
    """
    validation test to list all available quantities
    """
    @staticmethod
    def _save_quantities(catalog_name, quantities, filename):
        quantities = list(quantities)
        quantities.sort()
        with open(filename, 'w') as f:
            f.write('# ' + catalog_name + '\n')
            for q in quantities:
                f.write(str(q))
                f.write('\n')

    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        self._save_quantities(catalog_name, galaxy_catalog.list_all_quantities(), os.path.join(base_output_dir, 'quantities.txt'))
        self._save_quantities(catalog_name, galaxy_catalog.list_all_native_quantities(), os.path.join(base_output_dir, 'native_quantities.txt'))
        with open(os.path.join(base_output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(galaxy_catalog.get_catalog_info(), default_flow_style=False))
            f.write('\n')
        return TestResult(0, passed=True)
