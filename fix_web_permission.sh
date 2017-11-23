#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate python env
PYTHON="/global/common/cori/contrib/lsst/apps/anaconda/py3-envs/DESCQA/bin/python"

# import descqaweb to make sure it works
$PYTHON -E -c "import descqaweb"

# chmod
chmod o+r .htaccess
chmod o+rx . descqaweb/index.cgi
chmod -R o+rX descqaweb www

# end subshell
)
