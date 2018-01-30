from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne

from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['CheckQuantities']


def find_outlier(x):
    """
    return a bool array indicating outliers or not in *x*
    """
    l, m, h = np.percentile(x, [16.0, 50.0, 84.0])
    d = (h-l) * 0.5
    return (x > (m + d*3)) | (x < (m - d*3))


def calc_frac(x, func, total=None):
    """
    calculate the fraction of entries in *x* that satisfy *func*
    """
    total = total or len(x)
    return np.count_nonzero(func(x)) / total


def split_for_natural_sort(s):
    """
    split a string *s* for natural sort.
    """
    return tuple((int(y) if y.isdigit() else y for y in re.split(r'(\d+)', s)))


def evaluate_expression(expression, catalog_instance):
    """
    evaluate a numexpr expression on a GCR catalog
    """
    quantities_needed = set(ne.necompiler.precompile(expression)[-1])
    if not catalog_instance.has_quantities(quantities_needed):
        raise KeyError("Not all quantities needed exist")
    return ne.evaluate(expression,
                       local_dict=catalog_instance.get_quantities(quantities_needed),
                       global_dict={})


class CheckQuantities(BaseValidationTest):
    """
    Readiness test to check catalog quantities before image simulations
    """

    stats = OrderedDict((
        ('min', np.min),
        ('max', np.max),
        ('median', np.median),
        ('mean', np.mean),
        ('std', np.std),
        ('f_inf', np.isinf),
        ('f_nan', np.isnan),
        ('f_zero', np.logical_not),
        ('f_outlier', find_outlier),
    ))

    def __init__(self, **kwargs):
        self.quantities_to_check = kwargs.get('quantities_to_check', [])
        self.relations_to_check = kwargs.get('relations_to_check', [])
        assert (self.quantities_to_check or self.relations_to_check), 'must specify quantities_to_check or relations_to_check'
        assert all('quantities' in d for d in self.quantities_to_check), 'yaml file not correctly specified'
        self.nbins = kwargs.get('nbins', 50)
        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))
        super(CheckQuantities, self).__init__(**kwargs)


    def _format_row(self, quantity, plot_filename, results):
        output = ['<tr>', '<td title="{1}">{0}</td>'.format(quantity, plot_filename)]
        for s in self.stats:
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

        output_header.append('<span>Running readiness test on {} {}</span>'.format(catalog_name, getattr(catalog_instance, 'version', '')))

        for i, checks in enumerate(self.quantities_to_check):

            quantity_patterns = checks['quantities'] if isinstance(checks['quantities'], (tuple, list)) else [checks['quantities']]
            assert quantity_patterns, 'yaml file not specify correctly!'

            quantities_this = set()
            quantity_pattern = None
            for quantity_pattern in quantity_patterns:
                quantities_this.update(fnmatch.filter(all_quantities, quantity_pattern))

            if not quantities_this:
                output_header.append('<span class="fail">Found no matching quantities for {}</span>'.format(quantity_pattern))
                failed_count += 1
                continue

            quantities_this = sorted(quantities_this, key=split_for_natural_sort)

            if 'label' in checks:
                quantity_group_label = checks['label']
            else:
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

                value_finite = value[np.isfinite(value)]

                result_this_quantity = {}
                for s, func in self.stats.items():
                    if s == 'f_outlier':
                        s_value = calc_frac(value_finite, func, len(value))
                    elif s.startswith('f_'):
                        s_value = calc_frac(value, func)
                    else:
                        s_value = func(value_finite)

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

                quantity_hashes[tuple(result_this_quantity[s][0] for s in self.stats)].add(quantity)

                ax.hist(value_finite, self.nbins, histtype='step', fill=False, label=quantity, **next(self.prop_cycle))
                output_rows.append(self._format_row(quantity + (' [log]' if checks.get('log') else ''), plot_filename, result_this_quantity))

            ax.set_xlabel(('log ' if checks.get('log') else '') + quantity_group_label)
            ax.yaxis.set_ticklabels([])
            ax.set_title('{} {}'.format(catalog_name, getattr(catalog_instance, 'version', '')), fontsize='small')
            fig.tight_layout()
            if len(quantities_this) <= 9:
                leg = ax.legend(loc='best', fontsize='x-small', ncol=3, frameon=True, facecolor='white')
                leg.get_frame().set_alpha(0.5)
            fig.savefig(os.path.join(output_dir, plot_filename))
            plt.close(fig)

        for same_quantities in quantity_hashes.values():
            if len(same_quantities) > 1:
                output_header.append('<span class="fail">{} seem be to identical!</span>'.format(', '.join(same_quantities)))
                failed_count += 1

        for relation in self.relations_to_check:
            expr1, simeq, expr2 = relation.partition('~==')
            if simeq:
                func = lambda e1=expr1.strip(), e2=expr2.strip(): np.allclose(
                    evaluate_expression(e1, catalog_instance),
                    evaluate_expression(e2, catalog_instance),
                    equal_nan=True
                )
            else:
                func = lambda r=relation: evaluate_expression(r, catalog_instance).all()

            try:
                result = func()
            except Exception as e: # pylint: disable=broad-except
                output_header.append('<span class="fail">Not able to evaluate `{}`! {}</span>'.format(relation, e))
                failed_count += 1
                continue

            if result:
                output_header.append('<span>It is true that `{}`</span>'.format(relation))
            else:
                output_header.append('<span class="fail">`{}` not true!</span>'.format(relation))
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
            for s in self.stats:
                f.write('<td>{}</td>'.format(s))
            f.write('</tr></thead><tbody>\n')
            for line in output_rows:
                f.write(line)
                f.write('\n')
            f.write('</tbody></table></body></html>\n')

        return TestResult(passed=(failed_count == 0), score=failed_count)

