#!/usr/bin/env python
import os
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

form = cgi.FieldStorage()
targetFile  = form.getvalue("target_file")

print "<html><head></head><body>"
print "<img src=\"data:image/png;base64,%s\" width=\"100%%\">" % (open(targetFile, 'rb').read().encode("base64").replace("\n", ""))
print "</body></html>"

