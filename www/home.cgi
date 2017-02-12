#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import os
import sys
import stat
from utils.invocations_simple import Invocation, BigBoard
from utils import littleParser

# config
configDict = littleParser.parseFile('config')

pathToOutputDir = configDict['pathToOutputDir'] # must have
bigboard_cache = configDict.get('bigboard_cache')
siteTitle = configDict.get('siteTitle', '')
invocationsPerPage = int(configDict.get('invocationsPerPage', 25))
days_to_show = int(configDict.get('days_to_show', 15))

print '<!DOCTYPE html>'
print '<html>'
print '<body>'
print '<head>'
print '<title>{}</title>'.format(siteTitle)
print '<meta http-equiv="content-type" content="text/html; charset=utf-8">'
print '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
print '<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css">'
print '<link rel="stylesheet" href="style/style.css">'
print '</head>'
print '<body>'
print '<div id="pleasewait">'
print '<h1>Please wait while the table is being generated.</h1>'
print '</div>'
sys.stdout.flush()

form = cgi.FieldStorage()
try:
    this_page = int(form.getfirst('page', 1))
except:
    this_page = 1

bigboard = BigBoard(pathToOutputDir, bigboard_cache)
cache_dumped = bigboard.generate(days_to_show, bigboard_cache)

if cache_dumped:
    try:
        os.chmod(bigboard_cache, stat.S_IWOTH+stat.S_IROTH+stat.S_IWGRP+stat.S_IRGRP+stat.S_IRUSR+stat.S_IWUSR)
    except OSError:
        pass

# floating div which will be populated with the stats
# from one invocation when user hovers over a datestamp
print '<div id="statsWindow"><div id="statsHeader"></div><div id="statsBody"></div></div>'

# start main page
print '''<!DOCTYPE html>
<html>
<head>
<title>DESCQA: LSST DESC Quality Assurance</title>
<meta http-equiv="content-type" content="text/html; charset=utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/5.0.0/normalize.min.css">
<link rel="stylesheet" href="style/style.css">
</head>
<body>
<div id="header" class="without-secondary-menu"><div class="section clearfix" style="background-image: -webkit-linear-gradient(top, #dfe1e1 0%, #878d91 100%);">
<a href="index.cgi" title="Home" rel="home" id="logo">
<img src="http://lsst-desc.org/sites/default/files/desc-logo-small.png" alt="Home" />
</a>  
<div id="name-and-slogan">
<div id="site-name"> </div>
</div> 
</div></div>
'''
print '<a class="everblue" href="index.cgi">&lt; Back to home</a>'
print '<div class="title"><h1>{}</h1></div>'.format(siteTitle)

count = bigboard.get_count()

if not count:
    print '<h1>nothing to show!</h1>'
    print '</body></html>'
    sys.exit(0)


npages = ((count - 1) // invocationsPerPage) + 1
if this_page > npages:
    this_page = npages

print '<div id="pagesDiv">'

# print a "<<" (previous page link)
if this_page > 1:
    print '<a class="everblue" href="home.cgi?page={}">&lt;&lt;</a>'.format(this_page - 1)
else:
    print '<span style="color: gray">&lt;&lt;</span>'
print '&nbsp;|&nbsp;'
# print a ">>" (next page link)
if this_page < npages:
    print '<a class="everblue" href="home.cgi?page={}">&gt;&gt;</a>'.format(this_page + 1)
else:
    print '<span style="color: gray">&gt;&gt;</span>'
print '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'
page_links = []
for i in xrange(1, npages + 1):
    if i == this_page:
        page_links.append('<span style="color: gray">{}</span>'.format(i))
    else:
        page_links.append('<a class="everblue" href="./home.cgi?page={0}">{0}</a>'.format(i))

print '[&nbsp;{}&nbsp;]'.format('&nbsp;|&nbsp;'.join(page_links))
print '</div>'

print '<div class="legend">Legend:'
print '<img src="style/red.gif"/> Run did not successfully complete'
print '<img src="style/yellow.gif"/> Run completed but some tests have execution errors'
print '<img src="style/green.gif"/> All tests successfully completed (but may be skipped or may not pass the test)'
print '</div>'

print '<div>'
print bigboard.get_html(invocationsPerPage*(this_page-1), invocationsPerPage)
print '</div>'

print '<script src="style/statsWindow.js"></script>'
print '<script>'
print 'statsWindowInit();'
print 'document.getElementById("pleasewait").style.display="none";'
print '</script>'
print '</body>'
print '</html>'

