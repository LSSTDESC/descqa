from __future__ import unicode_literals
import cgi
from .interface import get_all_runs
from . import config

__all__ = ['prepare_bigtable']

try:
    unicode
except NameError:
    unicode = str


def format_status_count(status_count):
    output = []
    try:
        for name, d in status_count.items():
            total = sum(d.values())
            output.append(name + ' - ' + '; '.join(('{}/{} {}'.format(d[k], total, cgi.escape(k)) for k in d)))
    except AttributeError:
        if isinstance(status_count, unicode):
            output = [cgi.escape(l) for l in status_count.splitlines()]
    return '<br>'.join(output)


def format_bigtable_row(descqa_run):

    user = descqa_run.status.get('user', '')
    user = '&nbsp;({})'.format(user) if user else ''

    comment = descqa_run.status.get('comment', '')
    if len(comment) > 20:
        comment = comment[:20] + '...'
    if comment:
        comment = '<br>&nbsp;&nbsp;<i>{}</i>'.format(comment)

    test_status = format_status_count(descqa_run.status.get('status_count', {}))
    light = 'green'
    if not test_status:
        light = 'red'
        test_status = 'status file "STATUS.json" not found or cannot be read!'
    elif '_ERROR' in test_status:
        light = 'yellow'

    catalog_status = format_status_count(descqa_run.status.get('status_count_group_by_catalog', {}))

    output = []
    main_link = '&nbsp;<a href="index.cgi?run={}" onMouseOver="appear(\'{}\', \'{}\');" onMouseOut="disappear();">{}</a>'.format(\
            descqa_run.name, test_status, catalog_status, descqa_run.name)
    output.append('<td>{}{}{}</td>'.format(main_link, user, comment))
    output.append('<td><img src="style/{}.gif"></td>'.format(light))
    test_links = '&nbsp;|&nbsp;'.join(('<a href="index.cgi?run={0}&test={1}">{1}</a>'.format(descqa_run.name, t) for t in descqa_run.tests))
    catalog_links = '&nbsp;|&nbsp;'.join(('<a href="index.cgi?run={0}&catalog={1}">{1}</a>'.format(descqa_run.name, c) for c in descqa_run.catalogs))
    output.append('<td>TESTS:&nbsp;{}<br>{}{}&nbsp;</td>'.format(test_links, 'CATALOGS:&nbsp;' if catalog_links else '', catalog_links))

    return '\n'.join(output)


def prepare_bigtable(page=0):
    all_runs = get_all_runs(config.root_dir)
    n_per_page = config.run_per_page
    if all_runs:
        npages = ((len(all_runs) - 1) // n_per_page) + 1
        if page > npages:
            page = npages
        all_runs = all_runs[n_per_page*(page-1):n_per_page*page]

    table_out = []
    table_out.append('<table class="bigboard" border="0" width="100%" cellspacing="0">')
    for run in all_runs:
        table_out.append('<tr>{}</tr>'.format(format_bigtable_row(run)))
    table_out.append('</table>')

    return dict(table='\n'.join(table_out), page=page, npages=npages)
