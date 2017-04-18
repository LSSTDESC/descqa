#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()

import os
form = cgi.FieldStorage()
targetFile  = form.getfirst('target_file')

try:
    with open(targetFile, 'rb') as f:
        file_content = f.read()

except (OSError, IOError):
    print 'Content-type: text/plain\n'
    print '[Error] Cannot open/read file {}'.format(targetFile)
    
else:
    if targetFile.lower().endswith('.png'):
        print 'Content-type: text/html'
        print
        print '<!DOCTYPE html>'
        print '<html><body>'
        print '<img src="data:image/png;base64,{}" width="100%">'.format(file_content.encode('base64').replace('\n', ''))
        print '</body></html>'
    elif targetFile.lower().endswith('.pdf'):
        print 'Content-type: application/pdf'
        print 'Content-Length: {}'.format(len(file_content))
        print 'Content-Disposition: inline; filename="{}"'.format(os.path.basename(targetFile))
        print 
        print file_content
    else:
        print 'Content-type: text/plain'
        print 'Content-Length: {}'.format(len(file_content))
        print 
        print file_content

