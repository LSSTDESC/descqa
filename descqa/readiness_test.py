from __future__ import print_function, division, unicode_literals, absolute_import
import os
import re
import fnmatch
from itertools import cycle
from collections import defaultdict, OrderedDict
import numpy as np
import numexpr as ne
from scipy.stats import norm

from .base import BaseValidationTest, TestResult
from .plotting import plt


__all__ = ['CheckQuantities']


def check_uniqueness(x, mask=None):
    """ Return True if the elements of the input x are unique, else False.
    Optionally only evaluate uniqueness on a subset defined by the input mask.

    Examples
    --------
    >>> x = np.random.randint(0, 10, 100)
    >>> assert check_uniqueness(x) == False
    >>> assert check_uniqueness(np.arange(5)) == True
    """
    x = np.asarray(x)
    if mask is None:
        return x.size == np.unique(x).size
    else:
        return check_uniqueness(x[mask])


def find_outlier(x):
    """
    return a bool array indicating outliers or not in *x*
    """
    l, m, h = np.percentile(x, norm.cdf([-1, 0, 1])*100)
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


def check_relation(relation, catalog_instance):
    """
    check if *relation* is true in *catalog_instance*
    """
    expr1, simeq, expr2 = relation.partition('~==')

    if simeq:
        expr1 = expr1.strip()
        expr2 = expr2.strip()
        return np.allclose(
            evaluate_expression(expr1, catalog_instance),
            evaluate_expression(expr2, catalog_instance),
            equal_nan=True,
        )

    return evaluate_expression(relation, catalog_instance).all()


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
        self.uniqueness_to_check = kwargs.get('uniqueness_to_check', [])
        self.catalog_filters = kwargs.get('catalog_filters', [])
        self.lgndtitle_fontsize = kwargs.get('lgndtitle_fontsize', 10)
        
        if not any((
                self.quantities_to_check,
                self.relations_to_check,
                self.uniqueness_to_check,
        )):
            raise ValueError('must specify quantities_to_check, relations_to_check, or uniqueness_to_check')

        if not all(d.get('quantities') for d in self.quantities_to_check):
            raise ValueError('yaml file error: `quantities` must exist for each item in `quantities_to_check`')

        if not all(isinstance(d, str) for d in self.relations_to_check):
            raise ValueError('yaml file error: each item in `relations_to_check` must be a string')

        if not all(d.get('quantity') for d in self.uniqueness_to_check):
            raise ValueError('yaml file error: `quantity` must exist for each item in `uniqueness_to_check`')

        if self.catalog_filters:
            if not all(d.get('quantity') for d in self.catalog_filters):
                raise ValueError('yaml file error: `quantity` must exist for each item in `catalog_filters`')

            if not all(d.get('min') for d in self.catalog_filters) or all(d.get('max') for d in self.catalog_filters):
                raise ValueError('yaml file error: `min` or `max` must exist for each item in `catalog_filters`')
        
        self.enable_individual_summary = bool(kwargs.get('enable_individual_summary', True))
        self.enable_aggregated_summary = bool(kwargs.get('enable_aggregated_summary', False))
        self.always_show_plot = bool(kwargs.get('always_show_plot', True))

        self.nbins = int(kwargs.get('nbins', 50))
        self.prop_cycle = None

        self.current_catalog_name = None
        self.current_failed_count = None
        self._aggregated_header = list()
        self._aggregated_table = list()
        self._individual_header = list()
        self._individual_table = list()

        super(CheckQuantities, self).__init__(**kwargs)

    def record_result(self, results, quantity_name=None, more_info=None, failed=None, individual_only=False):
        if isinstance(results, dict):
            self.current_failed_count += sum(1 for v in results.values() if v[1] == 'fail')
        elif failed:
            self.current_failed_count += 1

        if self.enable_individual_summary:
            if quantity_name is None:
                self._individual_header.append(self.format_result_header(results, failed))
            else:
                self._individual_table.append(self.format_result_row(results, quantity_name, more_info))

        if self.enable_aggregated_summary and not individual_only:
            if quantity_name is None:
                results = '{} {}'.format(self.current_catalog_name, results) if self.current_catalog_name else results
                self._aggregated_header.append(self.format_result_header(results, failed))
            else:
                quantity_name = '{} {}'.format(self.current_catalog_name, quantity_name) if self.current_catalog_name else quantity_name
                self._aggregated_table.append(self.format_result_row(results, quantity_name, more_info))

    def format_result_row(self, results, quantity_name, more_info):
        more_info = 'title="{}"'.format(more_info) if more_info else ''
        output = ['<tr>', '<td {1}>{0}</td>'.format(quantity_name, more_info)]
        for s in self.stats:
            output.append('<td class="{1}" title="{2}">{0:.4g}</td>'.format(*results[s]))
        output.append('</tr>')
        return ''.join(output)

    @staticmethod
    def format_result_header(results, failed=False):
        return '<span {1}>{0}</span>'.format(results, 'class="fail"' if failed else '')

    def generate_summary(self, output_dir, aggregated=False):
        if aggregated:
            if not self.enable_aggregated_summary:
                return
            header = self._aggregated_header
            table = self._aggregated_table
        else:
            if not self.enable_individual_summary:
                return
            header = self._individual_header
            table = self._individual_table

        with open(os.path.join(output_dir, 'SUMMARY.html'), 'w') as f:
            f.write('<html><head><style>html{font-family: monospace;} table{border-spacing: 0;} thead,tr:nth-child(even){background: #ddd;} thead{font-weight: bold;} td{padding: 2px 8px;} .fail{color: #F00;} .none{color: #444;}</style></head><body>\n')

            f.write('<ul>\n')
            for line in header:
                f.write('<li>')
                f.write(line)
                f.write('</li>\n')
            f.write('</ul><br>\n')

            f.write('<table><thead><tr><td>Quantity</td>\n')
            for s in self.stats:
                f.write('<td>{}</td>'.format(s))
            f.write('</tr></thead><tbody>\n')
            for line in table:
                f.write(line)
                f.write('\n')
            f.write('</tbody></table></body></html>\n')

        if not aggregated:
            self._individual_header.clear()
            self._individual_table.clear()

    def run_on_single_catalog(self, catalog_instance, catalog_name, output_dir):

        all_quantities = sorted(map(str, catalog_instance.list_all_quantities(True)))

        self.prop_cycle = cycle(iter(plt.rcParams['axes.prop_cycle']))
        self.current_catalog_name = catalog_name
        self.current_failed_count = 0
        galaxy_count = None
        quantity_hashes = defaultdict(set)

        self.record_result('Running readiness test on {} {}'.format(
            catalog_name,
            getattr(catalog_instance, 'version', ''),
            individual_only=True,
        ))

        # check filters
        filters = []
        filter_labels = ''
        for d in self.catalog_filters:
            fq = d.get('quantity')
            if fq in all_quantities:
                filter_label=''
                qlabel = d.get('label') if d.get('label') else fq
                if d.get('min') is not None:
                    filters.append('{} >= {}'.format(fq, d.get('min')))
                    filter_label = '{} <= {}'.format(d.get('min'), qlabel)
                if d.get('max') is not None:
                    filters.append('{} < {}'.format(fq, d.get('max')))
                    flabel = '{} <= {}'.format(qlabel, d.get('max'))
                    filter_label = flabel if len(filter_label)==0 else re.sub(str(d.get('label')), flabel, filter_label)
                filter_labels = '$'+filter_label+'$' if len(filter_labels)==0 else ', '.join([filter_labels,
                                                                                              '$'+filter_label+'$'])
            else:
                self.record_result('Found no matching quantity for filtering on {}'.format(fq), failed=True)
                continue

        print(filters, filter_labels)

        for i, checks in enumerate(self.quantities_to_check):

            quantity_patterns = checks['quantities'] if isinstance(checks['quantities'], (tuple, list)) else [checks['quantities']]

            quantities_this = set()
            quantity_pattern = None
            for quantity_pattern in quantity_patterns:
                quantities_this.update(fnmatch.filter(all_quantities, quantity_pattern))

            if not quantities_this:
                self.record_result('Found no matching quantities for {}'.format(quantity_pattern), failed=True)
                continue

            quantities_this = sorted(quantities_this, key=split_for_natural_sort)

            if 'label' in checks:
                quantity_group_label = checks['label']
            else:
                quantity_group_label = re.sub('_+', '_', re.sub(r'\W+', '_', quantity_pattern)).strip('_')
            plot_filename = 'p{:02d}_{}.png'.format(i, quantity_group_label)

            fig, ax = plt.subplots()
            has_plot = False

            for quantity in quantities_this:
                if len(filters) > 0:
                    catalog_data = catalog_instance.get_quantities([quantity], filters=filters)
                    value = catalog_data[quantity]
                else:
                    value = catalog_instance[quantity]
                need_plot = False

                if galaxy_count is None:
                    galaxy_count = len(value)
                    self.record_result('Found {} entries in this catalog.'.format(galaxy_count))
                elif galaxy_count != len(value):
                    self.record_result('"{}" has {} entries (different from {})'.format(quantity, len(value), galaxy_count), failed=True)
                    need_plot = True

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
                        except (TypeError, ValueError):
                            flag |= (s_value != checks[s])
                        else:
                            if min_value is not None:
                                flag |= (s_value < min_value)
                            if max_value is not None:
                                flag |= (s_value > max_value)
                    else:
                        flag = None
                    result_this_quantity[s] = (
                        s_value,
                        'none' if flag is None else ('fail' if flag else 'pass'),
                        checks.get(s, ''),
                    )
                    if flag:
                        need_plot = True

                quantity_hashes[tuple(result_this_quantity[s][0] for s in self.stats)].add(quantity)
                self.record_result(
                    result_this_quantity,
                    quantity + (' [log]' if checks.get('log') else ''),
                    plot_filename
                )

                if need_plot or self.always_show_plot:
                    ax.hist(value_finite, self.nbins, histtype='step', fill=False, label=quantity, **next(self.prop_cycle))
                    has_plot = True

            if has_plot:
                ax.set_xlabel(('log ' if checks.get('log') else '') + quantity_group_label)
                ax.yaxis.set_ticklabels([])
                if checks.get('plot_min') is not None: #zero values fail otherwise
                    ax.set_xlim(left=checks.get('plot_min'))
                if checks.get('plot_max') is not None:
                    ax.set_xlim(right=checks.get('plot_max'))
                ax.set_title('{} {}'.format(catalog_name, getattr(catalog_instance, 'version', '')), fontsize='small')
                fig.tight_layout()
                if len(quantities_this) <= 9:
                    leg = ax.legend(loc='best', fontsize='x-small', ncol=3, frameon=True, facecolor='white',
                                    title=filter_labels, title_fontsize=self.lgndtitle_fontsize)
                    leg.get_frame().set_alpha(0.5)
                fig.savefig(os.path.join(output_dir, plot_filename))
            plt.close(fig)

        for same_quantities in quantity_hashes.values():
            if len(same_quantities) > 1:
                self.record_result('{} seem be to identical!'.format(', '.join(same_quantities)), failed=True)

        for relation in self.relations_to_check:
            try:
                result = check_relation(relation, catalog_instance)
            except Exception as e: # pylint: disable=broad-except
                self.record_result('Not able to evaluate `{}`! {}'.format(relation, e), failed=True)
                continue

            if result:
                self.record_result('It is true that `{}`'.format(relation))
            else:
                self.record_result('It is NOT true that `{}`'.format(relation), failed=True)

        for d in self.uniqueness_to_check:
            quantity = label = d.get('quantity')
            mask = d.get('mask')

            quantities_needed = [quantity]
            if mask is not None:
                quantities_needed.append(mask)
                label += '[{}]'.format(mask)

            if not catalog_instance.has_quantities(quantities_needed):
                self.record_result('{} does not exist'.format(' or '.join(quantities_needed)), failed=True)
                continue

            data = catalog_instance.get_quantities(quantities_needed)
            if check_uniqueness(data[quantity], data.get(mask)):
                self.record_result('{} is all unique'.format(label))
            else:
                self.record_result('{} has repeated entries!'.format(label), failed=True)

        self.generate_summary(output_dir)

        return TestResult(passed=(self.current_failed_count == 0), score=self.current_failed_count)

    def conclude_test(self, output_dir):
        self.generate_summary(output_dir, aggregated=True)
