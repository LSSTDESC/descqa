#!/usr/bin/env python
import os, sys
import cgi
sys.path.insert(0, "../lib")
import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()

targetDir = form.getvalue("target_dir")
targetTag = form.getvalue("target_tag")

if os.path.isfile("../config"):
  configDict = littleParser.parseFile("../config")
  siteTitle = configDict.get("siteTitle", [])

print "Content-type: text/html\n"
print "<html>"
print "<head>"
print "<title>%s</title>" % siteTitle
print open("style.css","r").read()
print "</head>"

print "<frameset cols=\"40%,*\">"
if targetTag:
  print "  <frame src=\"leftFrame.py?target_dir=%s#%s\" name=\"leftframe\">" % (targetDir, targetTag)
else:
  print "  <frame src=\"leftFrame.py?target_dir=%s\" name=\"leftframe\">" % targetDir
print "  <frame src=\"viewTextFile.py\" name=\"rightframe\">"
print "</frameset>"
print "</html>"
