#!/usr/bin/env python
import os, sys
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

sys.path.insert(0, "../lib")
import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()

targetDir = form.getvalue("target_dir")
targetTag = form.getvalue("target_tag")

try:
  configDict = littleParser.parseFile("../config")
  siteTitle = configDict.get("siteTitle", [])
except:
  siteTitle = ''

print "<html>"
print "<head>"
print "<title>%s</title>" % siteTitle
print open("style.css","r").read()
print "</head>"

print "<frameset cols=\"40%,*\">"
if targetTag:
  print "  <frame src=\"leftFrame.cgi?target_dir=%s#%s\" name=\"leftframe\">" % (targetDir, targetTag)
else:
  print "  <frame src=\"leftFrame.cgi?target_dir=%s\" name=\"leftframe\">" % targetDir
print "  <frame src=\"javascript:parent.blank()\" name=\"rightframe\">"
print "</frameset>"
print "</html>"
