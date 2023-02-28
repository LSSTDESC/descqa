from __future__ import division, unicode_literals, absolute_import
import os
from builtins import str #pylint: disable=W0622
import yaml
import numpy as np
import healpy as hp
from .base import BaseValidationTest, TestResult
from .plotting import plt

__all__ = ['ListAvailableQuantities', 'SkyArea']

class ListAvailableQuantities(BaseValidationTest):
    """
    validation test to list all available quantities
    """
    def __init__(self, **kwargs): #pylint: disable=W0231
        self.kwargs = kwargs
        self.calc_min_max = kwargs.get('calc_min_max', False)


    def _save_quantities(self, catalog_name, quantities, filename):
        is_dict = isinstance(quantities, dict)
        maxlen = max((len(q) for q in quantities))
        with open(filename, 'w') as f:
            if is_dict:
                f.write('{} {} {} \n'.format('# ' + catalog_name.ljust(maxlen-2), 'Minimum'.rjust(13), 'Maximum'.rjust(13)))
            else:
                f.write('# ' + catalog_name + '\n')
            for q in sorted(quantities):
                if is_dict:
                    f.write('{0} {1[0]:13.4g} {1[1]:13.4g} '.format(q.ljust(maxlen), quantities[q]))
                else:
                    f.write(str(q))
                f.write('\n')


    def _get_data_ranges(self, catalog_instance, native=False):

        quantities = catalog_instance.list_all_native_quantities() if native else catalog_instance.list_all_quantities()
        if not self.calc_min_max:
            return quantities

        if native:
            #check for name collisions and add native quantity
            quantities_needed = []
            gcr_quantities = catalog_instance.list_all_quantities()
            for q in quantities:
                if q in gcr_quantities:
                    catalog_instance.add_quantity_modifier(q + '_native', q)
                    quantities_needed.append(q + '_native')
                else:
                    quantities_needed.append(q)
        else:
            quantities_needed = quantities

        d_min = {}
        d_max = {}
        for data in catalog_instance.get_quantities(quantities_needed, return_iterator=True):
            for qx in quantities_needed:
                q = qx.replace('_native','') if qx.endswith('_native') else qx
                if data[qx].dtype.char in 'bBiulfd':
                    d_min[q] = min(np.nanmin(data[qx]), d_min.get(q, np.inf))
                    d_max[q] = max(np.nanmax(data[qx]), d_max.get(q, -np.inf))

        #clean_up q_native added quantities
        for qx in quantities_needed:
            if qx.endswith('_native'):
                catalog_instance.del_quantity_modifier(qx)

        return {q: (d_min[q], d_max[q]) for q in d_min}


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        self._save_quantities(catalog_name, self._get_data_ranges(catalog_instance, native=False), os.path.join(output_dir, 'quantities.txt'))
        self._save_quantities(catalog_name, self._get_data_ranges(catalog_instance, native=True), os.path.join(output_dir, 'native_quantities.txt'))
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(catalog_instance.get_catalog_info(), default_flow_style=False))
            f.write('\n')
        return TestResult(inspect_only=True)


class SkyArea(BaseValidationTest):
    """
    validation test to show sky area
    """
    def __init__(self, **kwargs): #pylint: disable=W0231
        self.nside = kwargs.get('nside', 64)
        assert hp.isnsideok(self.nside), '`nside` value {} not correct'.format(self.nside)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        if not catalog_instance.has_quantities(['ra_true', 'dec_true']):
            return TestResult(skipped=True)

        pixels = set()
        for d in catalog_instance.get_quantities(['ra_true', 'dec_true'], return_iterator=True):
            pixels.update(hp.ang2pix(self.nside, d['ra_true'], d['dec_true'], lonlat=True))

        frac = len(pixels) / hp.nside2npix(self.nside)
        skyarea = frac * np.rad2deg(np.rad2deg(4.0*np.pi))

        hp_map = np.empty(hp.nside2npix(self.nside))
        hp_map.fill(hp.UNSEEN)
        hp_map[list(pixels)] = 0

        hp.mollview(hp_map, title=catalog_name, coord='C', cbar=None)
        plt.savefig(os.path.join(output_dir, 'skymap.png'))
        plt.close()
        return TestResult(inspect_only=True, summary='approx. {:.7g} sq. deg.'.format(skyarea))
