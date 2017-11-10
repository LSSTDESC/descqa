from __future__ import print_function, unicode_literals
import cgi
from jinja2 import Environment, PackageLoader

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

    print('Content-type: text/html')
    print()

    if form.getfirst('header'):
        print(env.get_template('header.html').render())
        return

    run = form.getfirst('run', '')

    if run.lower() == 'all':
        try:
            page = int(form.getfirst('page', 1))
        except (ValueError, TypeError):
            page = 1
        print(env.get_template('bigtable.html').render(**prepare_bigtable(page)))

    elif run:
        catalog = form.getfirst('catalog')
        test = form.getfirst('test')

        if catalog or test:
            if form.getfirst('left'):
                print(env.get_template('leftpanel.html').render(**prepare_leftpanel(run, test, catalog)))
            else:
                print(env.get_template('twopanels.html').render(run=run, catalog=catalog, test=test))
        else:
            print(env.get_template('matrix.html').render(**prepare_matrix(run=run,
                    catalog_prefix=form.getfirst('catalog_prefix'),
                    test_prefix=form.getfirst('test_prefix'))))

    else:
        print(env.get_template('matrix.html').render(run=find_last_run()))
