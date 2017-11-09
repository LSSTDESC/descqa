import os
import stat
import json
from ..config import *
from .bigboard import BigBoard

__all__ = ['render', 'find_last_run']

def find_last_run():
    names = os.listdir(pathToOutputDir)
    names.sort(reverse=True)
    targetDir_base = names[0]
    for name in names:
        statusfile = os.path.join(pathToOutputDir, name, 'STATUS.json')
        if os.path.exists(statusfile):
            status = json.load(open(statusfile))
            if status.get('comment') == 'full run':
                targetDir_base = name
                break

    return targetDir_base


def render(template, page):
    bigboard = BigBoard(pathToOutputDir, bigboard_cache)
    cache_dumped = bigboard.generate(days_to_show, bigboard_cache, new_style=True)

    if cache_dumped:
        try:
            os.chmod(bigboard_cache, stat.S_IWOTH+stat.S_IROTH+stat.S_IWGRP+stat.S_IRGRP+stat.S_IRUSR+stat.S_IWUSR)
        except OSError:
            pass

    count = bigboard.get_count()
    if count:
        npages = ((count - 1) // invocationsPerPage) + 1
        if page > npages:
            page = npages
        html = bigboard.get_html(invocationsPerPage*(page-1), invocationsPerPage)
    else:
        html = '<h1>nothing to show!</h1>'

    return template.render(html=html, page=page, npages=npages)
