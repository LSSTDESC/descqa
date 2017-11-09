from __future__ import print_function
import cgi
from jinja2 import Environment, PackageLoader

from . import view_bigboard
from . import view_twopanels
from . import view_matrix

__all__ = ['run']

env = Environment(loader=PackageLoader('descqaweb', 'templates'))


def print_html_header():
    print('Content-type: text/html')
    print()


def run():
    form = cgi.FieldStorage()

    if form.getfirst('file'):
        view_twopanels.print_file(form.getfirst('file'))
        return

    print_html_header()

    if form.getfirst('header'):
        print(env.get_template('header.html').render())
        return

    run = form.getfirst('run', '')

    if run.lower() == 'all':
        try:
            page = int(form.getfirst('page', 1))
        except (ValueError, TypeError):
            page = 1
        print(view_bigboard.render(env.get_template('bigboard.html'), page))

    elif run:
        catalog = form.getfirst('catalog')
        test = form.getfirst('test')
        prefix = dict(catalog_prefix=form.getfirst('catalog_prefix'),
                      test_prefix=form.getfirst('test_prefix'))

        if catalog or test:
            if form.getfirst('left'):
                print(view_twopanels.render_left(env.get_template('leftpanel.html'), run, catalog, test, **prefix))
            else:
                print(view_twopanels.render(env.get_template('twopanels.html'), run, catalog, test, **prefix))
        else:
            print(view_matrix.render(env.get_template('matrix.html'), run, **prefix))

    else:
        run = view_bigboard.find_last_run()
        print(view_matrix.render(env.get_template('matrix.html'), run))
