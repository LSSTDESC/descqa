from __future__ import unicode_literals, print_function
import os
import sys
from . import config
from .interface import DescqaRun, encode_png

__all__ = ['prepare_leftpanel', 'print_file']

def prepare_leftpanel(run, test=None, catalog=None):

    try:
        descqa_run = DescqaRun(run, config.root_dir)
    except AssertionError:
        raise ValueError('Invalid run "{}"'.format(run))

    if test is None and catalog is None:
        raise ValueError('`test` and `catalog` cannot both be `None`')
    if test and test not in descqa_run.tests:
        raise ValueError('Invalid test {} for run {}'.format(test, run))
    if catalog and catalog not in descqa_run.catalogs:
        raise ValueError('Invalid catalog {} for run {}'.format(catalog, run))

    data = dict()
    data['run'] = descqa_run.name
    data['test'] = test
    data['catalog'] = catalog

    if test:
        data['group'] = [descqa_run[test, c] for c in descqa_run.catalogs]
        for item in data['group']:
            item.name = item.catalog
        data['summary'] = descqa_run[test]
        data['title'] = test
        data['is_group_by_catalog'] = False
    else:
        data['group'] = [descqa_run[t, catalog] for t in descqa_run.tests]
        for item in data['group']:
            item.name = item.test
        data['title'] = catalog
        data['summary'] = None
        data['is_group_by_catalog'] = True

    return data


def print_file(target_file, root_dir=config.root_dir):
    try:
        assert (not os.path.isabs(target_file)) or target_file.startswith(root_dir)

        with open(os.path.join(root_dir, target_file), 'rb') as f:
            file_content = f.read()

    except (OSError, IOError, AssertionError):
        print('Content-type: text/plain')
        print()
        print('[Error] Cannot open/read file {}'.format(target_file))

    else:
        if target_file.lower().endswith('.png'):
            print('Content-type: text/html')
            print()
            print('<!DOCTYPE html>')
            print('<html><body>')
            print('<img src="data:image/png;base64,{}" width="100%">'.format(encode_png(file_content)))
            print('</body></html>')
        elif target_file.lower().endswith('.pdf'):
            print('Content-type: application/pdf')
            print('Content-Length: {}'.format(len(file_content)))
            print('Content-Disposition: inline; filename="{}"'.format(os.path.basename(target_file)))
            print()
            sys.stdout.buffer.write(file_content)
        else:
            print('Content-type: text/plain')
            print('Content-Length: {}'.format(len(file_content)))
            print()
            sys.stdout.buffer.write(file_content)
