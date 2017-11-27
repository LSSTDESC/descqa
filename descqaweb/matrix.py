from __future__ import unicode_literals
import os
import time
import cgi
from . import config
from .interface import iter_all_runs, DescqaRun

__all__ = ['prepare_matrix']


def find_last_descqa_run():
    last_run = None
    for run in iter_all_runs(config.root_dir, 180):
        descqa_run = DescqaRun(run, config.root_dir, validated=True)
        if last_run is None:
            last_run = descqa_run
        if descqa_run.status.get('comment', '').strip().lower() == 'full run':
            last_run = descqa_run
            break
    return last_run


def format_filter_link(targetDir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
    text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
    if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
        return '<span style="color:gray">{}</span>'.format(text)
    new_test_prefix_str = '&test_prefix={}'.format(new_test_prefix) if new_test_prefix else ''
    new_catalog_prefix_str = '&catalog_prefix={}'.format(new_catalog_prefix) if new_catalog_prefix else ''
    return '<a href="?run={}{}{}">{}</a>'.format(targetDir, new_test_prefix_str, new_catalog_prefix_str, text)


def format_description(description_dict):
    output = []
    for k in sorted(description_dict):
        v = description_dict.get(k)
        if v:
            output.append('<tr><td>{}</td><td>{}</td></tr>'.format(k, v))
    return '\n'.join(output)


def prepare_matrix(run=None, catalog_prefix=None, test_prefix=None):

    if run:
        try:
            descqa_run = DescqaRun(run, config.root_dir)
        except AssertionError:
            raise ValueError('Invalid run "{}"'.format(run))
    else:
        descqa_run = find_last_descqa_run()

    data = dict()

    data['version_info'] = config.version_info
    data['run'] = descqa_run.name
    data['comment'] = descqa_run.status.get('comment', '')
    data['user'] = descqa_run.status.get('user', 'UNKNOWN')
    data['versions'] = ' | '.join(('{}: {}'.format(k, v) for k, v in descqa_run.status.get('versions', dict()).items()))

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

    catalogs_this = descqa_run.get_catalogs(catalog_prefix)

    table_width = (len(catalogs_this) + 1)*130
    if table_width > 1280:
        data['table_width'] = "100%"
    else:
        data['table_width'] = "{}px".format(table_width)

    matrix = list()
    matrix.append('<tr><td>&nbsp;</td>')
    for catalog in catalogs_this:
        matrix.append('<td><a href="?run={1}&catalog={0}">{0}</a></td>'.format(catalog, descqa_run.name))
    matrix.append('</tr>')
    for test in descqa_run.get_tests(test_prefix, True):
        matrix.append('<tr>')
        matrix.append('<td><a href="?run={0}&test={1}">{1}</a></td>'.format(descqa_run.name, test))
        for catalog in catalogs_this:
            item = descqa_run[test, catalog]
            matrix.append('<td class="{}"><a class="celllink" href="?run={}&test={}&catalog={}">{}<br>{}</a></td>'.format(\
                    item.status_color, descqa_run.name, test, catalog, item.status.rpartition('_')[-1], item.score))
        matrix.append('</tr>')
    data['matrix'] = '\n'.join(matrix)

    for type_this in ('Validation', 'Catalog'):
        key = '{}_description'.format(type_this.lower())
        if key in descqa_run.status:
            data[key] = '<table><thead><tr><td>{} Name</td><td>Description</td></tr></thead>\n<tbody>\n{}\n</tbody></table>'.format(
                type_this,
                format_description(descqa_run.status[key]),
            )

    return data
