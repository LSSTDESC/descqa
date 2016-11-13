#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()

form = cgi.FieldStorage()
targetFile  = form.getfirst('target_file')

if not targetFile:
    print 'Content-type: text/html\n'
elif targetFile.lower().endswith('.png'):
    print 'Content-type: text/html\n'
    print '<!DOCTYPE html>'
    print '<html><body>'
    print '<img src="data:image/png;base64,{}" width="100%">'.format(open(targetFile, 'rb').read().encode('base64').replace('\n', ''))
    print '</body></html>'
else:
    print 'Content-type: text/plain\n'
    print open(targetFile).read()

