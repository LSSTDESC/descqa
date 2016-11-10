#!/usr/bin/env python
import cgi, cgitb
cgitb.enable()
print "Content-type: text/html\n"

import os
import re
import time

config_dict = {}
with open('config') as f:
    for l in f:
        if l[0] != '#':
            k, has_colon, v = l.partition(':')
            if has_colon:
                config_dict[k.strip()] = v.strip()

# print the header section
print '<body>'
print '<head>'
print '<title>{0}</title>'.format(config_dict.get('siteTitle', ''))
print '<meta http-equiv="cache-control" content="no-cache">'
print '<meta http-equiv="Pragma" content="no-cache">'
print '<meta http-equiv="Expires" content="-1">'
print open("style.css","r").read()
print '<script src="lib/vanishPleaseWait.js"></script>'
print '<script src="lib/statsWindow.js"></script>'
print '<script src="lib/redirect.js"></script>'
print '</head>'

root_dir = config_dict['pathToOutdir']

# TODO: hack to get the actual path
if os.path.basename(root_dir) != 'edison':
    root_dir = os.path.join('edison')

now = time.time()
for i in xrange(5):
    time.strftime('%Y-%m-%d', time.localtime(now - 86400.0*i))

dates_available = set()
board = {}

for item in os.listdir(root_dir):
    full_path = os.path.join(root_dir, item)
    m = re.match(r'(20\d{2}-[01]\d-[0123]\d)(?:_(\d+))?', item)
    if m is None or not os.path.isdir(full_path):
        continue
    m = m.groups()
    if m[0] != date_wanted:
        dates_available.add(m[0])
        continue
    
    for subitem in os.listdir(full_path):
        if os.path.join(full_path, subitem)

    board[int(m[1] or 0)] = 
    

