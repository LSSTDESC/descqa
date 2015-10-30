#!/usr/bin/env python
import os, sys
import cgi
sys.path.insert(0, "../lib")
import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()

targetFile  = form.getvalue("target_file")
soughtBuild = form.getvalue("sought_build")

if targetFile:

  if os.path.isfile("../config"):
    configDict = littleParser.parseFile("../config")
    pathToOutdir = configDict.get("pathToOutdir", [])
    siteTitle = configDict.get("siteTitle", [])
  
  if (targetFile.find(pathToOutdir) == 0) and \
     (targetFile.find("/../") == -1):
  
    print "Content-type: text/html\n"
    print "\n" # DEV Why the hell does uncommenting above line cause
               # "page unavailable" error at my dad's house but just
               # a newline by itself (or anything except the correct 
               # header followed by a newline) allows it to work?!
    print "<html>"
    print "<head>"
    print "<title>%s</title>" % siteTitle
    print "<style type=\"text/css\">"
    print "body {font-family: Courier, Times, Helvetica, Geneva, Arial, sans-serif;"
    print "font-size: 12px;}"
    print "</style>"
    print "<script src=\"findText.js\"></script>"
    print "</head>"
    if soughtBuild:
      print "<body onLoad=\"javascript: findText('%s')\">" % soughtBuild
    else:
      print "<body>"
    print "<pre>"
    if targetFile:
      try:
        text = open(targetFile,"r").read()
      except Exception, e:
        print "Error opening file<br>"
        print e
      else:
        print text.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    print "</pre>"
    print "</body>"
    print "</html>"
  else:
    print "Content-type: text/html\n"
    print "<html><title></title><body>"
    print "Access denied"
    print "</body></html>"
else:
  print "Content-type: text/html\n"
  print "<html><title></title><body></body></html>"
