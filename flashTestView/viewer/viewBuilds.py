#!/usr/bin/env python
import sys, os
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import json
import re
sys.path.insert(0, "../lib")
import littleParser

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

    def get_html(self):
        if self.html is None:
            try:
                with open(os.path.join(self.path, 'STATUS')) as f:
                    status = f.readline().strip()
            except (OSError, IOError):
                status = 'NO_STATUS_FILE_ERROR'
            
            status = status.rpartition('_')[-1]
            color = color_dict.get(status, 'darkred')
            
            self.html = '<td style="background-color:{}">{}</td>'.format(color, cgi.escape(status))
        
        return self.html


class TestGroup(TestDir):
    members = None

    def get_members(self):
        if self.members is None:
            self.members = {}

            for name in os.listdir(self.path):
                try:
                    member = TestMember(name, self.path)
                except AssertionError:
                    continue
                self.members[name] = member
        return self.members

    def get_html(self, sorted_member_names):
        members = self.get_members()
        html = ['<td><a href="viewBuild.cgi?target_dir={}">{}</a></td>'.format(self.path, self.name)]
        for member_name in sorted_member_names:
            member = members.get(member_name)
            html.append(member.get_html() if member else '<td>&nbsp;</td>')
        return '<tr>{}</tr>'.format(''.join(html))


def get_filter_link(target_dir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
    text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
    if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
        return text
    return '<a href="viewBuilds.cgi?target_dir={}&test_prefix={}&catalog_prefix={}">{}</a>'.format(target_dir, new_test_prefix, new_catalog_prefix, text)


# -------------- form data ---------------- #
form = cgi.FieldStorage()
target_dir = form.getfirst('target_dir')
assert target_dir

target_dir = os.path.abspath(target_dir)
if not re.match(r'\d{4}-\d{2}-\d{2}', os.path.basename(target_dir)):
    print '<script>location.href="../home.cgi";</script>'
    sys.exit(0)

test_prefix = form.getfirst('test_prefix', '')
catalog_prefix = form.getfirst('catalog_prefix', '')

all_groups = []
for name in os.listdir(target_dir):
    try:
        group = TestGroup(name, target_dir)
    except AssertionError:
        continue
    all_groups.append(group)
all_groups.sort()

catalog_list= set()
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
    with open(os.path.join(target_dir, 'STATUS.json')) as f:
        master_status = json.load(f)
except:
    master_status = {}


try:
    configDict = littleParser.parseFile('../config')
    siteTitle = configDict.get('siteTitle', '')
except:
    siteTitle = ''

print '<html>'
print '<head>'
print '<meta http-equiv="content-type" content="text/html; charset=utf-8">'
print '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
print '<title>{}</title>'.format(siteTitle)
print '</head>'
print '''<style>
table {
    text-align: center;
    width: 100%;
    table-layout: fixed;
    font-size: 13px;
    font-family: verdana, helvetica;
    margin-top:20px;
}
h3 {margin: 0.2em 0;
    width:100%;
    background-color:grey;}
p{
    font-size: 12px;
    margin: 0;
}
td:first-child {
    text-align: right;
    width: 130px;
    font-weight: bold;
    padding-right: 0.4em;
}
tr {
    height: 35px;
}
tr:first-child{
    font-weight: bold;
}
</style>'''

print '<body>'


print '<p class="gohome"><a href="../home.cgi">Return to the list of all runs</a></p>'
print '<h3>{}</h3>'.format(os.path.basename(target_dir))
print '<p class="info">'
if master_status:
    print 'Run by {}.'.format(master_status.get('user', 'UNKNOWN'))
    time_used = master_status.get('end_time', -1.0) - master_status.get('start_time', 0.0)
    if time_used > 0:
        print 'This run took {:.1f} minute(s).'.format(time_used/60.0)
print '<br>&nbsp;</p>'

test_prefix_union.insert(0, '')
links = '&nbsp;|&nbsp;'.join((get_filter_link(target_dir, True, p, catalog_prefix, test_prefix, catalog_prefix) for p in test_prefix_union))
print '<p class="filter">Text prefix | {}</p>'.format(links)

catalog_prefix_union.insert(0, '')
links = '&nbsp;|&nbsp;'.join((get_filter_link(target_dir, False, test_prefix, p, test_prefix, catalog_prefix) for p in catalog_prefix_union))
print '<p class="filter">Catalog prefix | {}</p>'.format(links)

table_width = (len(catalog_list) + 1)*130
if table_width > 1000:
    table_width = "100%"
else:
    table_width = "{}px".format(table_width)

print '<div style="width:{}"><table>'.format(table_width)

header_row = ['<td><a href="viewBuild.cgi?target_dir={1}/{0}">{0}</a></td>'.format(name, os.path.join(target_dir, '_group_by_catalog')) for name in catalog_list]
print '<tr><td>&nbsp;</td>{}</tr>'.format('\n'.join(header_row))

for group in all_groups:
    if not test_prefix or group.prefix == test_prefix:
        print group.get_html(catalog_list)

print '</table></div>'
print '</body>'
print '</html>'

