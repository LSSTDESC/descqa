from __future__ import unicode_literals, print_function
import os
import sys
from . import config
from .interface import DescqaRun, b64encode

__all__ = ['prepare_leftpanel', 'print_file']

def prepare_leftpanel(run, test=None, catalog=None, right=None):

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
    data['right'] = right

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
        print('Content-Type: text/plain; charset=utf-8')
        print()
        sys.stdout.flush()
        print('[Error] Cannot open/read file {}'.format(target_file))

    else:
        if target_file.lower().endswith('.png'):
            print('Content-Type: text/html; charset=utf-8')
            print()
            sys.stdout.flush()
            print('<!DOCTYPE html>')
            print('<html><body>')
            print('<img src="data:image/png;base64,{}" width="100%">'.format(b64encode(file_content)))
            print('</body></html>')

        elif target_file.lower().endswith('.pdf'):
            print('Content-Type: application/pdf')
            print('Content-Length: {}'.format(len(file_content)))
            print('Content-Disposition: inline; filename="{}"'.format(os.path.basename(target_file)))
            print()
            sys.stdout.flush()
            try:
                sys.stdout.buffer.write(file_content)
            except AttributeError:
                print(file_content)

        elif target_file.lower().endswith('.html'):
            print('Content-Type: text/html; charset=utf-8')
            file_content = file_content.decode('utf-8')
            print('Content-Length: {}'.format(len(file_content)))
            print()
            sys.stdout.flush()
            print(file_content)

        else:
            print('Content-Type: text/plain; charset=utf-8')
            file_content = file_content.decode('utf-8')
            print('Content-Length: {}'.format(len(file_content)))
            print('Content-Disposition: inline; filename="{}"'.format(os.path.basename(target_file)))
            print()
            sys.stdout.flush()
            print(file_content)
