from __future__ import unicode_literals, absolute_import
import os
import numpy as np
import healpy as hp
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['SkyArea']

class SkyArea(BaseValidationTest):
    """
    validation test to show sky area
    """
    def __init__(self, nside=8, **kwargs):
        assert hp.isnsideok(nside), '`nside` value {} not correct'.format(nside)
        self._nside = nside

    def run_validation_test(self, galaxy_catalog, catalog_name, base_output_dir):
        if not galaxy_catalog.has_quantities(['ra_true', 'dec_true']):
            return TestResult(skipped=True)

        pixels = set()
        for d in galaxy_catalog.get_quantities(['ra_true', 'dec_true'], return_iterator=True):
            pixels.update(hp.ang2pix(self._nside, d['ra_true'], d['dec_true'], lonlat=True))

        hp_map = np.empty(hp.nside2npix(self._nside))
        hp_map.fill(hp.UNSEEN)
        hp_map[list(pixels)] = 0

        hp.mollview(hp_map, title=catalog_name, coord='C', cbar=None)
        plt.savefig(os.path.join(base_output_dir, 'skymap.png'))
        plt.close()
        return TestResult(0, passed=True)
