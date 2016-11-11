#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import os
import sys
import stat
sys.path.insert(0, 'lib')
from invocations_simple import Invocation, BigBoard
import littleParser

# config
pathToOutputDir = '/project/projectdirs/lsst/descqacmu/run/edison'
bigboard_cache = 'bigboard_cache.pkl'
invocationsPerPage = 25

try:
    configDict = littleParser.parseFile("config")
    siteTitle = configDict.get("siteTitle", '')
except:
    siteTitle = ''

# make sure website has write permissions in this folder
assert os.access(os.getcwd(), os.W_OK)

print '<html>'
print '<body>'
print '<head>'
print '<title>{}</title>'.format(siteTitle)

# next three lines ensure browsers don't cache, as caching can cause false
# appearances of the "please wait while the table is being regenerated" if
# the user uses the browser's "back" button.
print '<meta http-equiv="cache-control" content="no-cache">'
print '<meta http-equiv="Pragma" content="no-cache">'
print '<meta http-equiv="Expires" content="-1">'
print open("style.css","r").read()
print '<script src="lib/vanishPleaseWait.js"></script>'
print '<script src="lib/statsWindow.js"></script>'
print '<script src="lib/redirect.js"></script>'
print '</head>'
print "<body onLoad=\"vanishPleaseWait(); statsWindowInit()\">"
print "<div id=\"pleasewait\">"
print "DESCQA has generated new data since the last time this page was viewed.<br>"
print "Please wait while the table is being regenerated."
print "</div>"
sys.stdout.flush()

form = cgi.FieldStorage()
try:
    this_page = int(form.getvalue("page"))
except:
    this_page = 1

bigboard = BigBoard(pathToOutputDir)
try:
    bigboard.load(bigboard_cache)
except:
    pass

bigboard.generate()
try:
    bigboard.dump(bigboard_cache)
except:
    pass
else:
    os.chmod(bigboard_cache, stat.S_IWOTH+stat.S_IROTH+stat.S_IWGRP+stat.S_IRGRP+stat.S_IRUSR+stat.S_IWUSR)

# floating div which will be populated with the stats
# from one invocation when user hovers over a datestamp
print '<div id="statsWindow"><div id="statsHeader"></div><div id="statsBody"></div></div>'

# start main page
print '<div class="clearBlock">&nbsp;</div>'
print '<div id="titleDiv"><h1>{}</h1></div>'.format(siteTitle)


count = bigboard.get_count()

if not count:
    print '<h1>nothing to show!</h1>'
    print '</body></html>'
    sys.exit(0)


npages = ((count - 1) // invocationsPerPage) + 1
if this_page > npages:
    this_page = npages

print '<div class="clearBlock">&nbsp;</div>'
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
print '<div class="clearBlock">&nbsp;</div>'

print bigboard.get_html(invocationsPerPage*(this_page-1), invocationsPerPage)

print '</body>'
print '</html>'

