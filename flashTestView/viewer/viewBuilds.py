#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import os
import sys
import json
import re
import time

sys.path.insert(0, '..')
from utils import littleParser

color_dict = {'PASSED': 'green', 'SKIPPED': 'gold', 'FAILED': 'orangered', 'ERROR': 'darkred'}

class TestDir(object):
    def __init__(self, name, parent_dir,test_prefix,catalog_prefix):
        assert not name.startswith('_')
        self.name = name
        self.test_prefix=test_prefix
        self.catalog_prefix=catalog_prefix
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
                    lines = f.readlines()
            except (OSError, IOError):
                status = 'NO_STATUS_FILE_ERROR'
            else:
                status = lines[0].strip()
                try:
                    score = '<br>' + cgi.escape(lines[2].strip())
                except IndexError:
                    score = ''
            
            status = status.rpartition('_')[-1]
            color = color_dict.get(status, 'darkred')
            
            self.html = '<td style="background-color:{}"><a class="celllink" href="viewBuild.cgi?target_dir={}&target_item={}&test_prefix={}&catalog_prefix={}">{}{}</a></td>'.format(\
                    color, os.path.dirname(self.path), self.name,self.test_prefix, self.catalog_prefix, cgi.escape(status), score)
        
        return self.html


class TestGroup(TestDir):
    members = None

    def get_members(self):
        if self.members is None:
            self.members = {}

            for name in os.listdir(self.path):
                try:
                    member = TestMember(name, self.path, self.test_prefix, self.catalog_prefix)
                except AssertionError:
                    continue
                self.members[name] = member
        return self.members

    def get_html(self, sorted_member_names, target_dir_base=None):
        target_dir = self.path if target_dir_base is None else os.path.join(target_dir_base, self.name)
        members = self.get_members()
        html = ['<td><a href="viewBuild.cgi?target_dir={}&test_prefix={}&catalog_prefix={}">{}</a></td>'.format(target_dir,self.test_prefix,self.catalog_prefix,self.name)]
        for member_name in sorted_member_names:
            member = members.get(member_name)
            html.append(member.get_html() if member else '<td>&nbsp;</td>')
        return '<tr>{}</tr>'.format(''.join(html))


def get_filter_link(targetDir, istest, new_test_prefix, new_catalog_prefix, current_test_prefix, current_catalog_prefix):
    text = (new_test_prefix if istest else new_catalog_prefix) or 'CLEAR'
    if new_test_prefix == current_test_prefix and new_catalog_prefix == current_catalog_prefix:
        return '<span style="color:gray">{}</span>'.format(text)
    return '<a href="viewBuilds.cgi?target_dir={}&test_prefix={}&catalog_prefix={}">{}</a>'.format(targetDir, new_test_prefix, new_catalog_prefix, text)



# load config
try:
    configDict = littleParser.parseFile('../config')
except:
    configDict = {}

siteTitle = configDict.get('siteTitle', '')
pathToOutputDir = configDict.get('pathToOutputDir', '')
if not os.path.isabs(pathToOutputDir):
    raise ValueEror('`pathToOutputDir` in `config` should be an absolute path')


# check target_dir
form = cgi.FieldStorage()
targetDir = form.getfirst('target_dir', '')

targetDir = os.path.abspath(os.path.join(pathToOutputDir, targetDir))
targetDir_base = os.path.basename(targetDir)

if targetDir == pathToOutputDir:
    print '<script>location.href="../home.cgi";</script>'
    sys.exit(0)
elif targetDir_base == '_group_by_catalog':
    print '<script>location.href="viewBuilds.cgi?target_dir={}";</script>'.format(os.path.basename(os.path.dirname(targetDir)))
    sys.exit(0)
elif not re.match(r'\d{4}-\d\d-\d\d', targetDir_base):
    print '<script>location.href="viewBuild.cgi?target_dir={}";</script>'.format(targetDir)
    sys.exit(0)


test_prefix = form.getfirst('test_prefix', '')
catalog_prefix = form.getfirst('catalog_prefix', '')

all_groups = []
for name in os.listdir(targetDir):
    try:
        group = TestGroup(name, targetDir, test_prefix, catalog_prefix)
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
    with open(os.path.join(targetDir, 'STATUS.json')) as f:
        master_status = json.load(f)
except:
    master_status = {}


try:
    configDict = littleParser.parseFile('../config')
    siteTitle = configDict.get('siteTitle', '')
except:
    siteTitle = ''

print '<!DOCTYPE html>'
print '<html>'
print '<head>'
print '<title>{}</title>'.format(siteTitle)
print '<meta http-equiv="content-type" content="text/html; charset=utf-8">'
print '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
print '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css">'
print '<link rel="stylesheet" href="../style/style.css">'
print '</head>'
print '<body>'

print '<a class="everblue" href="../home.cgi">&lt; Back to "big table" (list of all runs)</a></p>'
print '<div class="title"><h1>{}</h1></div>'.format(targetDir_base)
if master_status:
    print '<div class="runinfo">'
    comment =  master_status.get('comment', '')
    if comment:
        print comment, '<br>'
    print 'Run initiated by {}'.format(master_status.get('user', 'UNKNOWN')),
    if 'start_time' in master_status:
        print time.strftime('at %Y/%m/%d %H:%M:%S PT', time.localtime(master_status.get('start_time'))),
    print '<br>'
    time_used = master_status.get('end_time', -1.0) - master_status.get('start_time', 0.0)
    if time_used > 0:
        print 'This run took {:.1f} minute(s).'.format(time_used/60.0)
    print '</div>'


print '<hr><div class="nav">'
test_prefix_union.insert(0, '')
links = '&nbsp;|&nbsp;'.join((get_filter_link(targetDir_base, True, p, catalog_prefix, test_prefix, catalog_prefix) for p in test_prefix_union))
print '[&nbsp;Test prefix: {}&nbsp;]<br>'.format(links)

catalog_prefix_union.insert(0, '')
links = '&nbsp;|&nbsp;'.join((get_filter_link(targetDir_base, False, test_prefix, p, test_prefix, catalog_prefix) for p in catalog_prefix_union))
print '[&nbsp;Catalog prefix: {}&nbsp;]'.format(links)
print '</div><hr>'


table_width = (len(catalog_list) + 1)*130
if table_width > 1280:
    table_width = "100%"
else:
    table_width = "{}px".format(table_width)

print '<div style="width:{}"><table class="matrix">'.format(table_width)

header_row = ['<td><a href="viewBuild.cgi?target_dir={1}/{0}&test_prefix={2}&catalog_prefix={3}">{0}</a></td>'.format(name, os.path.join(targetDir_base, '_group_by_catalog'),test_prefix,catalog_prefix) for name in catalog_list]
print '<tr><td>&nbsp;</td>{}</tr>'.format('\n'.join(header_row))

for group in all_groups:
    if not test_prefix or group.prefix == test_prefix:
        print group.get_html(catalog_list, targetDir_base)

print '</table></div>'

print '</body>'
print '</html>'

