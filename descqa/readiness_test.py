from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
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

    stats = {
        'max': np.max,
        'min': np.min,
        'mean': np.mean,
        'std': np.std,
        'median': np.median,
        'finite_frac': None,
        'outlier_frac': lambda x: calc_frac(x, find_outlier),
    }


    def __init__(self, **kwargs):
        self.quantities_to_check = kwargs['quantities_to_check']
        self.nbins = kwargs.get('nbins', 50)
        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))


    def _format_row(self, quantity, results):
        output = ['<tr>', '<td>{}</td>'.format(quantity)]
        for s in self.stats:
            output.append('<td class="{}">{:.4g}</td>'.format(*results[s]))
        output.append('</tr>')
        return ''.join(output)


    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        all_quantities = sorted(catalog_instance.list_all_quantities(True))

        failed_count = 0
        output_rows = []
        output_header = []
        existing_filenames = []

        for quantity_pattern, checks in self.quantities_to_check.items():
            quantities_this = fnmatch.filter(all_quantities, quantity_pattern)

            if not quantities_this:
                output_header.append('<p class="fail">Found no matching quantities for {}</p>'.format(quantity_pattern))
                failed_count += 1
                continue

            filename = re.sub(r'\W+', '_', quantity_pattern).strip('_')
            while filename in existing_filenames:
                filename += '_'
            existing_filenames.append(filename)

            quantities_this = catalog_instance.get_quantities(quantities_this)

            fig, ax = plt.subplots()

            for quantity, value in quantities_this.items():
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
                for s, func in self.stats.items():
                    s_value = finite_frac if s == 'finite_frac' else func(value)
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
                    result_this_quantity[s] = ('none' if flag is None else ('fail' if flag else 'pass'), s_value)
                    if flag:
                        failed_count += 1

                ax.hist(value, self.nbins, histtype='step', fill=False, label=quantity, **next(self.prop_cycle))
                output_rows.append(self._format_row(quantity, result_this_quantity))

            ax.set_ylabel('log ' if checks.get('log') else '' + filename)
            ax.legend()
            ax.set_title('{} {}'.format(catalog_name, getattr(catalog_instance, 'version', '')))
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, filename+'.png'))
            plt.close(fig)

        with open(os.path.join(output_dir, 'results.html'), 'w') as f:
            f.write('<html><head><style>html{font-family: monospace;} tr{padding: 4px;} .fail{color: #F00;} .none{color: #555;}</style></head><body>\n')

            for line in output_header:
                f.write(line)
                f.write('\n')

            f.write('<table><thead><td>Quantity</td>\n')
            for s in self.stats:
                f.write('<td>{}</td>'.format(s))
            f.write('\n')
            for line in output_rows:
                f.write(line)
                f.write('\n')
            f.write('</table></body></html>\n')

        return TestResult(passed=(failed_count==0), score=failed_count)
