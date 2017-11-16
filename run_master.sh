#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# set output directory
DESCQAROOTDIR="/global/projecta/projectdirs/lsst/groups/CS/descqa"
OUTPUTDIR="$DESCQAROOTDIR/run/v2"

# activate python env
export PYTHONPATH=""
PYTHON="/global/common/cori/contrib/lsst/apps/anaconda/py3-envs/DESCQA/bin/python"
# PYTHON="/global/common/software/lsst/common/miniconda/py2-4.3.21/bin/python"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
$PYTHON master.py "$OUTPUTDIR" "$@"

# end subshell
)
