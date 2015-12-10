#!/usr/bin/env python
import os, sys
import cgi
import tempfile
sys.path.insert(0, "../lib")
import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()

targetFile  = form.getvalue("target_file")

if os.path.isfile("../config"):
  configDict = littleParser.parseFile("../config")
  pathToOutdir = configDict.get("pathToOutdir", [])
  siteTitle = configDict.get("siteTitle", [])

if (targetFile.find(pathToOutdir) == 0) and \
   (targetFile.find("/../") == -1) and \
   (len(targetFile) >= 4) and \
   (targetFile[-4:] == ".png"):

  print "Content-type: text/html\n"
  print "<html><title></title><body>"
  if targetFile:
    try:
      print "<img src=\"data:image/png;base64,{0}\" width=\"100%\">".format(open(targetFile, 'rb').read().encode("base64").replace("\n", ""))
    except Exception, e:
      print "Error opening file<br>"
      print e
  print "</body></html>"

#  print "Content-type: image/png\n"
#  if targetFile:
#    try:
#      print open(targetFile, 'rb').read()
#    except Exception, e:
#      print "Error opening file<br>"
#      print e
else:
  print "Content-type: text/html\n"
  print "<html><title></title><body>"
  print "Access denied"
  print "</body></html>"
