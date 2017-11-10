#!/bin/bash
set -e

/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/bin/python -c "import descqaweb"

chmod o+rx descqaweb descqaweb/templates style index.cgi . ..
chmod -R o+r .

