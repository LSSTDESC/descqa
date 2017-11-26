#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate python env
PYTHON="/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/bin/python"

# set output directory
OUTPUTDIR="/global/projecta/projectdirs/lsst/groups/CS/descqa/run/v1"
URL="https://portal.nersc.gov/project/lsst/descqa/v1/"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import descqarun; descqarun.main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" -p "descqa_v1" -w "$URL" "$@"

# end subshell
)
