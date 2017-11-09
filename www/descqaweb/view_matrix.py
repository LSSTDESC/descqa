from __future__ import unicode_literals
import os
import time
import json
import cgi
from .config import *
from .utils import cmp

__all__ = ['render']

color_dict = {'PASSED': 'green', 'SKIPPED': 'gold', 'FAILED': 'orangered', 'ERROR': 'darkred'}

class TestDir(object):
    def __init__(self, name, parent_dir):
        assert not name.startswith('_')
        self.name = name
        self.path = os.path.join(parent_dir, name)
        assert os.path.isdir(self.path)
        self.prefix = name.partition('_')[0]

    def __cmp__(self, other):
        return cmp(self.name, other.name)


class TestMember(TestDir):
    html = None
    def __init__(self, name, parent_dir, groupname):
        super(TestMember,self).__init__(name, parent_dir)
        self.groupname = groupname


    def get_html(self):
        if self.html is None:
            try:
                with open(os.path.join(self.path, 'STATUS')) as f:
                    lines = f.readlines()
                status = lines[0].strip()
            except (OSError, IOError, IndexError):
                status = 'NO_STATUS_FILE_ERROR'
                lines = []

            try:
                score = '<br>' + cgi.escape(lines[2].strip())
            except IndexError:
                score = ''

            status = status.rpartition('_')[-1]
            color = color_dict.get(status, 'darkred')

            self.html = '<td class="{}"><a class="celllink" href="index.cgi?run={}&test={}&catalog={}">{}{}</a></td>'.format(color, os.path.basename(os.path.dirname(os.path.dirname(self.path))), self.groupname, self.name, cgi.escape(status), score)

        return self.html


class TestGroup(TestDir):
    members = None

    def get_members(self):
        if self.members is None:
            self.members = {}

            for name in os.listdir(self.path):
                try:
                    member = TestMember(name, self.path, self.name)
                except AssertionError:
                    continue
                self.members[name] = member
        return self.members

    def get_html(self, sorted_member_names, target_dir_base=None):
        target_dir_base = target_dir_base or self.path
        members = self.get_members()
        html = ['<td><a href="index.cgi?run={0}&test={1}">{1}</a></td>'.format(target_dir_base, self.name)]
        for member_name in sorted_member_names:
            member = members.get(member_name)
            html.append(member.get_html() if member else '<td>&nbsp;</td>')
        return '<tr>{}</tr>'.format(''.join(html))


def get_filter_link(targetDir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
    text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
    if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
        return '<span style="color:gray">{}</span>'.format(text)
    return '<a href="index.cgi?run={}&test_prefix={}&catalog_prefix={}">{}</a>'.format(targetDir, new_test_prefix, new_catalog_prefix, text)


def render(template, run, catalog_prefix=None, test_prefix=None):

    targetDir_base = run
    targetDir = os.path.abspath(os.path.join(pathToOutputDir, run))

    all_groups = []
    for name in os.listdir(targetDir):
        try:
            group = TestGroup(name, targetDir)
        except AssertionError:
            continue
        all_groups.append(group)
    all_groups.sort()

    catalog_list = set()
    test_prefix_union = set()
    catalog_prefix_union = set()

    for group in all_groups:
        test_prefix_union.add(group.prefix)
        for member in group.get_members().itervalues():
            catalog_prefix_union.add(member.prefix)
            if not catalog_prefix or member.prefix == catalog_prefix:
                catalog_list.add(member.name)

    catalog_list = sorted(catalog_list)
    test_prefix_union = sorted(test_prefix_union)
    catalog_prefix_union = sorted(catalog_prefix_union)

    try:
        with open(os.path.join(targetDir, 'STATUS.json')) as f:
            master_status = json.load(f)
    except Exception:
        master_status = {}

    comment = None
    user = None
    start_time = None
    time_used = None

    if master_status:
        comment = master_status.get('comment', '')
        user = master_status.get('user', 'UNKNOWN')
        if 'start_time' in master_status:
            start_time = time.strftime('at %Y/%m/%d %H:%M:%S PT', time.localtime(master_status.get('start_time')))
        time_used = (master_status.get('end_time', -1.0) - master_status.get('start_time', 0.0))/60.0

    test_prefix_union.insert(0, '')
    links = '&nbsp;|&nbsp;'.join((get_filter_link(run, True, p, catalog_prefix, test_prefix, catalog_prefix) for p in test_prefix_union))
    test_links_str = '[&nbsp;Test prefix: {}&nbsp;]<br>'.format(links)

    catalog_prefix_union.insert(0, '')
    links = '&nbsp;|&nbsp;'.join((get_filter_link(run, False, test_prefix, p, test_prefix, catalog_prefix) for p in catalog_prefix_union))
    catalog_links_str = '[&nbsp;Catalog prefix: {}&nbsp;]'.format(links)

    table_width = (len(catalog_list) + 1)*130
    if table_width > 1280:
        table_width = "100%"
    else:
        table_width = "{}px".format(table_width)

    header_row = ['<td><a href="index.cgi?run={1}&catalog={0}&test_prefix={2}&catalog_prefix={3}">{0}</a></td>'.format(name, targetDir_base, test_prefix, catalog_prefix) for name in catalog_list]
    header_row_str = '<tr><td>&nbsp;</td>{}</tr>'.format('\n'.join(header_row))

    return template.render(comment=comment, user=user, start_time=start_time,
                           time_used=time_used, test_links=test_links_str,
                           catalog_links=catalog_links_str, table_width=table_width,
                           header_row=header_row_str, all_groups=all_groups,
                           catalog_list=catalog_list, targetDir=targetDir_base,
                           test_prefix=test_prefix, catalog_prefix=catalog_prefix)
