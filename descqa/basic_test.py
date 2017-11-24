from __future__ import unicode_literals, absolute_import
import os
from builtins import str
import yaml
import healpy as hp
from .base import BaseValidationTest, TestResult

__all__ = ['ListAvailableQuantities', 'SkyArea']

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


class SkyArea(BaseValidationTest):
    """
    validation test to show sky area
    """
    def __init__(self, **kwargs):
        self.nside = kwargs.get('nside', 16)
        assert hp.isnsideok(self.nside), '`nside` value {} not correct'.format(self.nside)


    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        if not galaxy_catalog.has_quantities(['ra_true', 'dec_true']):
            return TestResult(skipped=True)

        pixels = set()
        for d in galaxy_catalog.get_quantities(['ra_true', 'dec_true'], return_iterator=True):
            pixels.update(hp.ang2pix(self.nside, d['ra_true'], d['dec_true'], lonlat=True))

        hp_map = np.empty(hp.nside2npix(self.nside))
        hp_map.fill(hp.UNSEEN)
        hp_map[list(pixels)] = 0

        hp.mollview(hp_map, title=catalog_name, coord='C', cbar=None)
        plt.savefig(os.path.join(base_output_dir, 'skymap.png'))
        plt.close()
        return TestResult(0, passed=True)
