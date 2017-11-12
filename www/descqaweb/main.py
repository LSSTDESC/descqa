from __future__ import print_function, unicode_literals
import cgi
from jinja2 import Environment, PackageLoader

from . import config
from .bigtable import *
from .twopanels import *
from .matrix import *

__all__ = ['run']

env = Environment(loader=PackageLoader('descqaweb', 'templates'))

def run():
    form = cgi.FieldStorage()

    if form.getfirst('file'):
        print_file(form.getfirst('file'))
        return

    print('Content-Type: text/html; charset=utf-8')
    print()

    if form.getfirst('header'):
        print(env.get_template('header.html').render(full_header=True, please_wait=False, siteTitle=config.site_title))
        return

    _run = form.getfirst('run', '')

    if _run.lower() == 'all':
        try:
            page = int(form.getfirst('page', 1))
        except (ValueError, TypeError):
            page = 1
        print(env.get_template('header.html').render(full_header=True, please_wait=True, siteTitle=config.site_title))
        print(env.get_template('bigtable.html').render(**prepare_bigtable(page)))
        return

    elif _run:
        catalog = form.getfirst('catalog')
        test = form.getfirst('test')

        if catalog or test:
            if form.getfirst('left'):
                print(env.get_template('header.html').render(full_header=False, please_wait=True, siteTitle=config.site_title))
                print(env.get_template('leftpanel.html').render(**prepare_leftpanel(_run, test, catalog)))
            else:
                print(env.get_template('twopanels.html').render(run=_run, catalog=catalog, test=test))
            return

    print(env.get_template('header.html').render(full_header=True, please_wait=True, siteTitle=config.site_title))
    print(env.get_template('matrix.html').render(**prepare_matrix(
        run=_run,
        catalog_prefix=form.getfirst('catalog_prefix'),
        test_prefix=form.getfirst('test_prefix'),
    )))
