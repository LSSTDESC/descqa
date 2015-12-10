#!/usr/bin/env python
import os, sys
import cgi, tempfile
sys.path.insert(0, "../lib")
import littleParser

# -------------- form data ---------------- #
form = cgi.FieldStorage()
targetFile = form.getvalue("target_file")

if os.path.isfile("../config"):
  configDict = littleParser.parseFile("../config")
  pathToOutdir = configDict.get("pathToOutdir", [])
  siteTitle = configDict.get("siteTitle", [])

if (targetFile.find(pathToOutdir) == 0) and \
   (targetFile.find("/../") == -1):

  print "Content-type: text/html\n"
  print "<html>"
  print "<head>"
  
  if targetFile:
    if os.path.isfile(targetFile):
      tmpDir = tempfile.mkdtemp(dir="../tmp")
      linkDest = os.path.normpath(os.path.join("../tmp", tmpDir, os.path.basename(targetFile)))
      os.symlink(targetFile, linkDest)
      print "<meta http-equiv=\"refresh\" content=\"0; url=%s\">" % linkDest
      print "</head>"
    else:
      print "<title>%s</title>" % siteTitle
      print "</head>"
      print "<body>"
      print "%s does not exist or is not a file" % targetFile
      print "</body>"
  
  else:
    print "<title>%s</title>" % siteTitle
    print "<style>"
    print "body {font-family: Courier, Times, Helvetica, Geneva, Arial, sans-serif;"
    print "font-size: 12px;}"
    print "</style>"
    print "</head>"
    print "<body>"
    print "no target file specified."
    print "</body>"
  
  print "</html>"
else:
  print "Content-type: text/html\n"
  print "<html><title></title><body>"
  print "Access denied"
  print "</body></html>"
