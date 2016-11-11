#!/usr/bin/env python
import os
import cgi, cgitb
cgitb.enable()
print "Content-type: text/plain\n"

form = cgi.FieldStorage()
targetFile  = form.getvalue("target_file")
with open(targetFile) as f:
    print f.read()

