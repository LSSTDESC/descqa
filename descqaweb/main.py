from __future__ import print_function, unicode_literals
import sys
import cgi
import html
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
        print_file(html.escape(form.getfirst('file')))
        return

    print('Content-Type: text/html; charset=utf-8')
    print()
    sys.stdout.flush()

    if form.getfirst('header'):
        print(env.get_template('header.html').render(full_header=True, header_page=True, config=config))
        return

    _run = html.escape(form.getfirst('run', ''))

    if _run.lower() == 'all':
        page = _convert_to_integer(form.getfirst('page'), 1)
        months = _convert_to_integer(form.getfirst('months'), config.months_to_search)
        search = {item: html.escape(form.getfirst(item)) for item in ('users', 'tests', 'catalogs') if form.getfirst(item)}
        print(env.get_template('header.html').render(full_header=True, please_wait=True, config=config))
        sys.stdout.flush()
        print(env.get_template('bigtable.html').render(**prepare_bigtable(page, months, search)))
        return

    elif _run:
        catalog = html.escape(form.getfirst('catalog'))
        test = html.escape(form.getfirst('test'))

        if catalog or test:
            if form.getfirst('left'):
                print(env.get_template('header.html').render(please_wait=True, config=config))
                sys.stdout.flush()
                print(env.get_template('leftpanel.html').render(**prepare_leftpanel(_run, test, catalog)))
            else:
                print(env.get_template('twopanels.html').render(run=_run, catalog=catalog, test=test, right=html.escape(form.getfirst('right'))))
            return

    print(env.get_template('header.html').render(full_header=True, please_wait=True, config=config))
    sys.stdout.flush()
    if _run or getattr(config, 'use_latest_run_as_home', True):
        print(env.get_template('matrix.html').render(**prepare_matrix(
            run=_run,
            catalog_prefix=html.escape(form.getfirst('catalog_prefix')),
            test_prefix=html.escape(form.getfirst('test_prefix')),
        )))
    else:
        print(env.get_template('home.html').render(general_info=config.general_info))
