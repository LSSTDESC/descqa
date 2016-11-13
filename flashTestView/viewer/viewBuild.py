#!/usr/bin/env python
import sys
import cgi, cgitb
cgitb.enable()
print 'Content-type: text/html\n'

sys.path.insert(0, '..')
from utils import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()

targetDir = form.getfirst('target_dir')
assert targetDir

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
print '</head>'
print '<frameset cols="50%,*">'
print '<frame src="leftFrame.cgi?target_dir={}" name="leftframe">'.format(targetDir)
print '<frame src="javascript:parent.blank()" name="rightframe">'
print '</frameset>'
print '</html>'
