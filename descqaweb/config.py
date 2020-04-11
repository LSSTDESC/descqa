from __future__ import unicode_literals

__all__ = ['site_title', 'root_dir', 'general_info', 'static_dir', 'run_per_page', 'logo_filename', 'github_url', 'months_to_search']

root_dir = '/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2'
site_title = 'DESCQA (v2): LSST DESC Quality Assurance for Galaxy Catalogs'

run_per_page = 20
months_to_search = 3

static_dir = 'web-static'
logo_filename = 'desc-logo-small.png'
github_url = 'https://github.com/lsstdesc/descqa'

general_info = '''
This is DESCQA v2. You can also visit the previous version, <a class="everblue" href="https://portal.nersc.gov/projecta/lsst/descqa/v1/">DESCQA v1</a>.
<br><br>
The DESCQA framework executes validation tests on mock galaxy catalogs.
These tests and catalogs are contributed by LSST DESC collaborators.
See <a href="https://arxiv.org/abs/1709.09665" target="_blank">the DESCQA paper</a> for more information.
Full details about the catalogs and tests, and how to contribute, are available <a href="https://confluence.slac.stanford.edu/x/Z0uKDQ" target="_blank">here</a> (collaborators only).
The source code of DESCQA is hosted in <a href="https://github.com/LSSTDESC/descqa/" target="_blank">this GitHub repo</a>.
'''

use_latest_run_as_home = False
