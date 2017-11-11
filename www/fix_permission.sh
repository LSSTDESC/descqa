#!/bin/bash
set -e

/global/common/cori/contrib/lsst/apps/anaconda/py3-envs/DESCQA/bin/python -c "import descqaweb"

chmod o+rx descqaweb descqaweb/templates style index.cgi . ..
chmod -R o+r .

