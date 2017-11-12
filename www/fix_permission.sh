#!/bin/bash
set -e

/global/common/cori/contrib/lsst/apps/anaconda/py3-envs/DESCQA/bin/python -c "import descqaweb"

chmod o+rx index.cgi
chmod -R o+rX .
chmod o+r ../index.html
chmod o+rx ..

