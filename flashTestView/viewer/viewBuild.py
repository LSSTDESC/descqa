#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print 'Content-type: text/html\n'

import os
import sys
import re

sys.path.insert(0, '..')
from utils import littleParser

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
elif re.match(r'\d{4}-\d\d-\d\d', targetDir_base):
    print '<script>location.href="viewBuilds.cgi?target_dir={}";</script>'.format(targetDir_base)
    sys.exit(0)
elif targetDir_base == '_group_by_catalog':
    print '<script>location.href="viewBuilds.cgi?target_dir={}";</script>'.format(os.path.basename(os.path.dirname(targetDir)))
    sys.exit(0)

# start printing webpage
print '<!DOCTYPE html>'
print '<html>'
print '<head>'
print '<title>{}</title>'.format(siteTitle)
print '<meta http-equiv="content-type" content="text/html; charset=utf-8">'
print '</head>'
print '<frameset cols="50%,*">'
print '<frame src="leftFrame.cgi?target_dir={}" name="leftframe">'.format(targetDir)
print '<frame src="javascript:parent.blank()" name="rightframe">'
print '</frameset>'
print '</html>'

