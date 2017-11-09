from __future__ import unicode_literals
import os
import re
import time
import cgi
import json
from ..utils import cmp

try:
    unicode
except NameError:
    unicode = str

__all__ = ['Invocation']

class Invocation:
    """
    encapsulates data visible for a single FlashTest invocation, (a
    date, possibly with a suffix) at the top level of FlashTestView
    """
    def __init__(self, name, dir_path, days_to_show=None):
        self.path = os.path.join(dir_path, name)
        assert os.path.isdir(self.path)
        assert not os.path.exists(os.path.join(self.path, '.lock'))
        self.name = name
        m = re.match(r'(20\d{2}-[01]\d-[0123]\d)(?:_(\d+))?', self.name)
        assert m is not None
        m = m.groups()
        self.date = m[0]
        self.sameday_index = int(m[1] or 0)
        if days_to_show is not None:
            assert self.date >= time.strftime('%Y-%m-%d', time.localtime(time.time()-86400.0*days_to_show))
        self.keep = True
        self.html = None

    def __cmp__(self, other):
        return cmp(other.date, self.date) or cmp(other.sameday_index, self.sameday_index)

    @staticmethod
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

    def gen_invocation_html(self, new_style):
        tests = [d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and not d.startswith('_')]
        tests.sort()

        catalogs = []
        if os.path.isdir(os.path.join(self.path, '_group_by_catalog')):
            catalogs = [d for d in os.listdir(os.path.join(self.path, '_group_by_catalog'))]
            catalogs.sort()

        try:
            with open(os.path.join(self.path, 'STATUS.json')) as f:
                master_status = json.load(f)
        except (IOError, OSError):
            master_status = {}

        user = master_status.get('user', '')
        user = '&nbsp;({})'.format(user) if user else ''

        comment = master_status.get('comment', '')
        if len(comment) > 20:
            comment = comment[:20] + '...'
        if comment:
            comment = '<br>&nbsp;&nbsp;<i>{}</i>'.format(comment)

        test_status = self.format_status_count(master_status.get('status_count', {}))
        light = 'green'
        if not test_status:
            light = 'red'
            test_status = 'status file "STATUS.json" not found or cannot be read!'
        elif '_ERROR' in test_status:
            light = 'yellow'

        catalog_status = self.format_status_count(master_status.get('status_count_group_by_catalog', {}))

        output = []
        if new_style:
            main_link = '&nbsp;<a href="index.cgi?run={}" onMouseOver="appear(\'{}\', \'{}\');" onMouseOut="disappear();">{}</a>'.format(\
                    self.name, test_status, catalog_status, self.name)
            output.append('<td>{}{}{}</td>'.format(main_link, user, comment))
            output.append('<td><img src="style/{}.gif"></td>'.format(light))
            test_links = '&nbsp;|&nbsp;'.join(('<a href="index.cgi?run={0}&test={1}">{1}</a>'.format(self.name, t) for t in tests))
            catalog_links = '&nbsp;|&nbsp;'.join(('<a href="index.cgi?run={0}&catalog={1}">{1}</a>'.format(self.name, c) for c in catalogs))
        else:
            main_link = '&nbsp;<a href="viewer/viewBuilds.cgi?target_dir={}" onMouseOver="appear(\'{}\', \'{}\');" onMouseOut="disappear();">{}</a>'.format(\
                    self.name, test_status, catalog_status, self.name)
            output.append('<td>{}{}{}</td>'.format(main_link, user, comment))
            output.append('<td><img src="style/{}.gif"></td>'.format(light))
            test_links = '&nbsp;|&nbsp;'.join(('<a href="viewer/viewBuild.cgi?target_dir={0}/{1}">{1}</a>'.format(self.name, t) for t in tests))
            catalog_links = '&nbsp;|&nbsp;'.join(('<a href="viewer/viewBuild.cgi?target_dir={0}/_group_by_catalog/{1}">{1}</a>'.format(self.name, c) for c in catalogs))
        output.append('<td>TESTS:&nbsp;{}<br>{}{}&nbsp;</td>'.format(test_links, 'CATALOGS:&nbsp;' if catalog_links else '',  catalog_links))

        self.html = '\n'.join(output)
