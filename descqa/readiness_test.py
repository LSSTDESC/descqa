from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict
import numpy as np

from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['CheckQuantities']


def find_outlier(x):
    l, m, h = np.percentile(x, [16.0, 50.0, 84.0])
    d = (h-l) * 0.5
    return (x > (m + d*3)) | (x < (m - d*3))


def calc_frac(x, func):
    return np.count_nonzero(func(x))/len(x)


class CheckQuantities(BaseValidationTest):
    """
    Readiness test to check catalog quantities before image simulations
    """

    stats_keys = ('min', 'max', 'median', 'mean', 'std', 'finite_frac', 'outlier_frac')

    def __init__(self, **kwargs):
        self.quantities_to_check = kwargs['quantities_to_check']
        assert all('quantities' in d for d in self.quantities_to_check), 'yaml file not correctly specified'
        self.nbins = kwargs.get('nbins', 50)
        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))


    def _format_row(self, quantity, plot_filename, results):
        output = ['<tr>', '<td title="{1}">{0}</td>'.format(quantity, plot_filename)]
        for s in self.stats_keys:
            output.append('<td class="{1}" title="{2}">{0:.4g}</td>'.format(*results[s]))
        output.append('</tr>')
        return ''.join(output)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        all_quantities = sorted(catalog_instance.list_all_quantities(True))

        galaxy_count = None
        failed_count = 0
        output_rows = []
        output_header = []
        quantity_hashes = defaultdict(set)

        for i, checks in enumerate(self.quantities_to_check):

            quantity_patterns = checks['quantities'] if isinstance(checks['quantities'], (tuple, list)) else [checks['quantities']]
            quantities_this = set()
            for quantity_pattern in quantity_patterns:
                quantities_this.update(fnmatch.filter(all_quantities, quantity_pattern))

            if not quantities_this:
                output_header.append('<span class="fail">Found no matching quantities for {}</span>'.format(quantity_pattern))
                failed_count += 1
                continue

            quantity_group_label = re.sub('_+', '_', re.sub(r'\W+', '_', quantity_pattern)).strip('_')
            plot_filename = 'p{:02d}_{}.png'.format(i, quantity_group_label)
            fig, ax = plt.subplots()

            for quantity in quantities_this:
                value = catalog_instance[quantity]

                if galaxy_count is None:
                    galaxy_count = len(value)
                    output_header.append('<span>Found {} entries in this catalog.</span>'.format(galaxy_count))
                elif galaxy_count != len(value):
                    output_header.append('<span class="fail">"{}" has {} entries (different from {})</span>'.format(quantity, len(value), galaxy_count))
                    failed_count += 1

                if checks.get('log'):
                    value = np.log10(value)

                finite_mask = np.isfinite(value)
                if finite_mask.any():
                    value = value[finite_mask]
                    finite_frac = np.count_nonzero(finite_mask) / len(finite_mask)
                else:
                    finite_frac = 1.0
                del finite_mask

                result_this_quantity = {}
                for s in self.stats_keys:
                    if s == 'finite_frac':
                        s_value = finite_frac
                    elif s == 'outlier_frac':
                        s_value = calc_frac(value, find_outlier)
                    else:
                        s_value = getattr(np, s)(value)

                    flag = False
                    if s in checks:
                        try:
                            min_value, max_value = checks[s]
                            if min_value is not None:
                                flag |= (s_value < min_value)
                            if max_value is not None:
                                flag |= (s_value > max_value)
                        except TypeError:
                            flag |= (s_value != checks[s])
                    else:
                        flag = None
                    result_this_quantity[s] = (
                        s_value,
                        'none' if flag is None else ('fail' if flag else 'pass'),
                        checks.get(s, ''),
                    )
                    if flag:
                        failed_count += 1

                quantity_hashes[tuple(result_this_quantity[s][0] for s in self.stats_keys)].add(quantity)

                ax.hist(value, self.nbins, histtype='step', fill=False, label=quantity, **next(self.prop_cycle))
                output_rows.append(self._format_row(quantity, plot_filename, result_this_quantity))

            ax.set_xlabel(('log ' if checks.get('log') else '') + quantity_group_label)
            ax.yaxis.set_ticklabels([])
            ax.set_title('{} {}'.format(catalog_name, getattr(catalog_instance, 'version', '')), fontsize='small')
            fig.tight_layout()
            ax.legend(loc='best', fontsize='small')
            fig.savefig(os.path.join(output_dir, plot_filename))
            plt.close(fig)

        for same_quantities in quantity_hashes.values():
            if len(same_quantities) > 1:
                output_header.append('<span class="fail">{} seem be to identical!</span>'.format(', '.join(same_quantities)))
                failed_count += 1

        with open(os.path.join(output_dir, 'SUMMARY.html'), 'w') as f:
            f.write('<html><head><style>html{font-family: monospace;} table{border-spacing: 0;} thead,tr:nth-child(even){background: #ddd;} thead{font-weight: bold;} td{padding: 2px 8px;} .fail{color: #F00;} .none{color: #444;}</style></head><body>\n')

            f.write('<ul>\n')
            for line in output_header:
                f.write('<li>')
                f.write(line)
                f.write('</li>\n')
            f.write('</ul><br>\n')

            f.write('<table><thead><tr><td>Quantity</td>\n')
            for s in self.stats_keys:
                f.write('<td>{}</td>'.format(s))
            f.write('</tr></thead><tbody>\n')
            for line in output_rows:
                f.write(line)
                f.write('\n')
            f.write('</tbody></table></body></html>\n')

        return TestResult(passed=(failed_count == 0), score=failed_count)
