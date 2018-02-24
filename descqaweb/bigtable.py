from __future__ import unicode_literals
import html
from .interface import iter_all_runs, DescqaRun
from . import config

__all__ = ['prepare_bigtable']

try:
    unicode
except NameError:
    unicode = str  #pylint: disable=redefined-builtin


def format_status_count(status_count):
    output = []
    try:
        for name, d in status_count.items():
            total = sum(d.values())
            output.append(name + ' - ' + '; '.join(('{}/{} {}'.format(d[k], total, html.escape(k)) for k in d)))
    except AttributeError:
        if isinstance(status_count, unicode):
            output = [html.escape(l) for l in status_count.splitlines()]
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
    main_link = '&nbsp;<a href="?run={}" onMouseOver="appear(\'{}\', \'{}\');" onMouseOut="disappear();">{}</a>'.format(\
            descqa_run.name, test_status, catalog_status, descqa_run.name)
    output.append('<td>{}{}{}</td>'.format(main_link, user, comment))
    output.append('<td><img src="{}/{}.gif"></td>'.format(config.static_dir, light))
    test_links = '&nbsp;|&nbsp;'.join(('<a href="?run={0}&test={1}">{1}</a>'.format(descqa_run.name, t) for t in descqa_run.tests))
    catalog_links = '&nbsp;|&nbsp;'.join(('<a href="?run={0}&catalog={1}">{1}</a>'.format(descqa_run.name, c) for c in descqa_run.catalogs))
    output.append('<td>TESTS:&nbsp;{}<br>{}{}&nbsp;</td>'.format(test_links, 'CATALOGS:&nbsp;' if catalog_links else '', catalog_links))

    return '\n'.join(output)


def filter_search_results(descqa_run, search):
    if 'users' in search and descqa_run.status.get('user') not in search['users'].split():
        return False
    if 'tests' in search and not all(any(t.startswith(ts) for t in descqa_run.tests) for ts in search['tests'].split()):
        return False
    if 'catalogs' in search and not all(any(c.startswith(cs) for c in descqa_run.catalogs) for cs in search['catalogs'].split()):
        return False
    return True


def prepare_bigtable(page=1, months=3, search=None):
    all_runs = list(iter_all_runs(config.root_dir, months_to_search=months))
    if search:
        all_runs = [descqa_run
                    for descqa_run in (DescqaRun(run, config.root_dir, validated=True) for run in all_runs)
                    if filter_search_results(descqa_run, search)]

    n_per_page = config.run_per_page
    npages = ((len(all_runs) - 1) // n_per_page) + 1
    if page > npages:
        page = npages
    all_runs = all_runs[n_per_page*(page-1):n_per_page*page]

    if not search:
        all_runs = [DescqaRun(run, config.root_dir, validated=True) for run in all_runs]

    table_out = []
    table_out.append('<table class="bigboard" border="0" width="100%" cellspacing="0">')
    for run in all_runs:
        table_out.append('<tr>{}</tr>'.format(format_bigtable_row(run)))
    table_out.append('</table>')

    return dict(table='\n'.join(table_out), page=page, npages=npages, static_dir=config.static_dir, search=search)
