from __future__ import print_function, unicode_literals
import sys
import cgi
from jinja2 import Environment, PackageLoader

from . import config
from .bigtable import prepare_bigtable
from .twopanels import prepare_leftpanel, print_file
from .matrix import prepare_matrix

__all__ = ['run']

env = Environment(loader=PackageLoader('descqaweb', 'templates'))


def _convert_to_integer(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def run():
    form = cgi.FieldStorage()

    if form.getfirst('file'):
        print_file(form.getfirst('file'))
        return

    print('Content-Type: text/html; charset=utf-8')
    print()
    sys.stdout.flush()

    if form.getfirst('header'):
        print(env.get_template('header.html').render(full_header=True, header_page=True, config=config))
        return

    _run = form.getfirst('run', '')

    if _run.lower() == 'all':
        page = _convert_to_integer(form.getfirst('page'), 1)
        months = _convert_to_integer(form.getfirst('months'), config.months_to_search)
        search = {item: form.getfirst(item) for item in ('users', 'tests', 'catalogs') if form.getfirst(item)}
        print(env.get_template('header.html').render(full_header=True, please_wait=True, config=config))
        sys.stdout.flush()
        print(env.get_template('bigtable.html').render(**prepare_bigtable(page, months, search)))
        return

    elif _run:
        catalog = form.getfirst('catalog')
        test = form.getfirst('test')

        if catalog or test:
            if form.getfirst('left'):
                print(env.get_template('header.html').render(please_wait=True, config=config))
                sys.stdout.flush()
                print(env.get_template('leftpanel.html').render(**prepare_leftpanel(_run, test, catalog)))
            else:
                print(env.get_template('twopanels.html').render(run=_run, catalog=catalog, test=test, right=form.getfirst('right')))
            return

    print(env.get_template('header.html').render(full_header=True, please_wait=True, config=config))
    sys.stdout.flush()
    print(env.get_template('matrix.html').render(**prepare_matrix(
        run=_run,
        catalog_prefix=form.getfirst('catalog_prefix'),
        test_prefix=form.getfirst('test_prefix'),
    )))
