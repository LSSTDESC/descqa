from __future__ import unicode_literals, absolute_import
import os
from builtins import str
import yaml
import numpy as np
import healpy as hp
from .base import BaseValidationTest, TestResult
from .plotting import plt
from collections import defaultdict

__all__ = ['ListAvailableQuantities', 'SkyArea']

class ListAvailableQuantities(BaseValidationTest):
    """
    validation test to list all available quantities
    """
    def __init__(self, **kwargs):
        
        self.kwargs = kwargs
        self.calc_min_max = kwargs.get('calc_min_max', False)

    def _save_quantities(self, catalog_name, quantities, filename):
        quantities = list(quantities)
        quantities.sort()
        maxlen = max((len(q) for q in quantities))
        qtype = 'native' if filename.find('native')!=-1 else 'GCR'
        with open(filename, 'w') as f:
            if self.calc_min_max:
                f.write('{} {} {} \n'.format('# ' + catalog_name.ljust(maxlen), 'Minimum'.rjust(13), 'Maximum'.rjust(13)))
            else:
                f.write('# ' + catalog_name + '\n')
            for q in quantities:
                if self.calc_min_max:
                    f.write('{} {:13.4g} {:13.4g} '.format(q.ljust(maxlen), self.ranges[qtype + '-min'].get(q, np.nan), self.ranges[qtype + '-max'].get(q, np.nan)))
                else:
                    f.write(str(q))
                f.write('\n')

    def _get_data_ranges(self, catalog_instance):

        self.ranges = defaultdict(dict)

        d_min = self.ranges['GCR-min']
        d_max = self.ranges['GCR-max']
        quantities = catalog_instance.list_all_quantities()
        for data in catalog_instance.get_quantities(quantities, return_iterator=True):
            for q in quantities:
                if data[q].dtype.char in 'bBiulfd':
                    d_min[q] = min(np.nanmin(data[q]), d_min.get(q, np.inf))
                    d_max[q] = max(np.nanmax(data[q]), d_max.get(q, -np.inf))
                
        native_quantities = catalog_instance.list_all_native_quantities()
        quantities_needed=[]
        #check for name collisions and add native quantity
        for q in native_quantities:
            if q in quantities:
                catalog_instance.add_quantity_modifier(q + '_native', q)
                quantities_needed.append(q + '_native')
            else:
                quantities_needed.append(q)

        d_min = self.ranges['native-min']
        d_max = self.ranges['native-max']
        for data in catalog_instance.get_quantities(quantities_needed, return_iterator=True):
            for qx in quantities_needed:
                q = qx.replace('_native','')
                if data[qx].dtype.char in 'bBiulfd':
                    d_min[q] = min(np.nanmin(data[qx]), d_min.get(q, np.inf))
                    d_max[q] = max(np.nanmax(data[qx]), d_max.get(q, -np.inf))

        #clean_up q_native added quantities
        for qx in quantities_needed:
            if qx.find('_native')!=-1:
                catalog_instance.del_quantity_modifier(qx)

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        if self.calc_min_max:
            self._get_data_ranges(catalog_instance)
        self._save_quantities(catalog_name, catalog_instance.list_all_quantities(), os.path.join(output_dir, 'quantities.txt'))
        self._save_quantities(catalog_name, catalog_instance.list_all_native_quantities(), os.path.join(output_dir, 'native_quantities.txt'))
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(catalog_instance.get_catalog_info(), default_flow_style=False))
            f.write('\n')
        return TestResult(0, passed=True)


class SkyArea(BaseValidationTest):
    """
    validation test to show sky area
    """
    def __init__(self, **kwargs):
        self.nside = kwargs.get('nside', 16)
        assert hp.isnsideok(self.nside), '`nside` value {} not correct'.format(self.nside)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):
        if not catalog_instance.has_quantities(['ra_true', 'dec_true']):
            return TestResult(skipped=True)

        pixels = set()
        for d in catalog_instance.get_quantities(['ra_true', 'dec_true'], return_iterator=True):
            pixels.update(hp.ang2pix(self.nside, d['ra_true'], d['dec_true'], lonlat=True))

        hp_map = np.empty(hp.nside2npix(self.nside))
        hp_map.fill(hp.UNSEEN)
        hp_map[list(pixels)] = 0

        hp.mollview(hp_map, title=catalog_name, coord='C', cbar=None)
        plt.savefig(os.path.join(output_dir, 'skymap.png'))
        plt.close()
        return TestResult(0, passed=True)
