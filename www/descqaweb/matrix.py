from __future__ import unicode_literals
import os
import time
import cgi
from . import config
from .interface import DescqaRun

__all__ = ['prepare_matrix']


def format_filter_link(targetDir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
    text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
    if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
        return '<span style="color:gray">{}</span>'.format(text)
    new_test_prefix_str = '&test_prefix={}'.format(new_test_prefix) if new_test_prefix else ''
    new_catalog_prefix_str = '&catalog_prefix={}'.format(new_catalog_prefix) if new_catalog_prefix else ''
    return '<a href="index.cgi?run={}{}{}">{}</a>'.format(targetDir, new_test_prefix_str, new_catalog_prefix_str, text)


def prepare_matrix(run, catalog_prefix=None, test_prefix=None):

    try:
        descqa_run = DescqaRun(run, config.root_dir)
    except AssertionError:
        raise ValueError('Invalid run "{}"'.format(run))

    data = dict()

    data['run'] = descqa_run.name
    data['comment'] = descqa_run.status.get('comment', '')
    data['user'] = descqa_run.status.get('user', 'UNKNOWN')

    if 'start_time' in descqa_run.status:
        data['start_time'] = time.strftime('at %Y/%m/%d %H:%M:%S PT', time.localtime(descqa_run.status.get('start_time')))
        data['time_used'] = (descqa_run.status.get('end_time', -1.0) - descqa_run.status.get('start_time', 0.0))/60.0
    else:
        data['start_time'] = None
        data['time_used'] = -1.0

    links = '&nbsp;|&nbsp;'.join((format_filter_link(run, True, p, catalog_prefix, test_prefix, catalog_prefix) \
            for p in ('',) + descqa_run.test_prefixes))
    data['test_links'] = '[&nbsp;Test prefix: {}&nbsp;]<br>'.format(links)

    links = '&nbsp;|&nbsp;'.join((format_filter_link(run, False, test_prefix, p, test_prefix, catalog_prefix) \
            for p in ('',) + descqa_run.catalog_prefixes))
    data['catalog_links'] = '[&nbsp;Catalog prefix: {}&nbsp;]'.format(links)

    table_width = (len(descqa_run.catalogs) + 1)*130
    if table_width > 1280:
        data['table_width'] = "100%"
    else:
        data['table_width'] = "{}px".format(table_width)

    matrix = list()
    matrix.append('<tr><td>&nbsp;</td>')
    for catalog in descqa_run.get_catalogs(catalog_prefix, True):
        matrix.append('<td><a href="index.cgi?run={1}&catalog={0}">{0}</a></td>'.format(catalog, descqa_run.name))
    matrix.append('</tr>')
    for test in descqa_run.get_tests(test_prefix, True):
        matrix.append('<tr>')
        matrix.append('<td><a href="index.cgi?run={0}&test={1}">{1}</a></td>'.format(descqa_run.name, test))
        for catalog in descqa_run.get_catalogs(catalog_prefix, True):
            item = descqa_run[test, catalog]
            matrix.append('<td class="{}"><a class="celllink" href="index.cgi?run={}&test={}&catalog={}">{}<br>{}</a></td>'.format(\
                    item.status_color, descqa_run.name, test, catalog, item.status.rpartition('_')[-1], item.score))
        matrix.append('</tr>')
    data['matrix'] = '\n'.join(matrix)

    return data
