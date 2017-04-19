#!/usr/bin/env python
import cgi
import os

form = cgi.FieldStorage()
target_dir = form.getfirst('target_dir')
target_file = form.getfirst('target_file')

dirs = [
    ('hlists', 'MBII DMO halo catalogs', '/project/projectdirs/lsst/www/MassiveBlack/hlists'), 
    ('trees', 'MBII DMO merger trees', '/project/projectdirs/lsst/www/MassiveBlack/trees'),
]

if target_dir and target_file:
    for key, name, this_dir in dirs:
        if target_dir == key:
            file_path = os.path.join(this_dir, target_file)
            if os.path.isfile(file_path):
                break
    else:
        file_path = None
else:
    file_path = None


if file_path:
    import sys
    import shutil
    size = os.path.getsize(file_path)
    print 'Content-Type: application/octet-stream'
    print 'Content-Disposition: attachment; filename={}'.format(target_file)
    print 'Content-Length: {}'.format(size)
    print 
    with open(file_path, 'rb') as f_in:
        shutil.copyfileobj(f_in, sys.stdout)


else:
    print 'Content-type: text/html\n'
    with open('descqa/templates/header.html') as f:
        print f.read()

    print '<h2>If you download these catalogs, please <a href="mailto:heitmann@anl.gov?subject=Use DESCQA MBII DMO catalogs&cc=yymao.astro@gmail.com">send an email</a> to nofify the DESCQA team (Katrin Heitman and Yao-Yuan Mao)</h2>'

    for key, name, this_dir in dirs:
        files = os.listdir(this_dir)
        if not files:
            continue

        files.sort()
        print '<h2>{}</h2>'.format(name)
        print '<div class="catalogList">'
        for f in files:
            print '<a href="./catalogs.cgi?target_dir={0}&target_file={1}">{1}</a>'.format(key, f)
        print '</div>'

