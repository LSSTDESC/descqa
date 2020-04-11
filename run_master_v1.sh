#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate python env
PYTHON="/global/common/cori/contrib/lsst/apps/anaconda/py2-envs/DESCQA/bin/python"
export KCORRECT_DIR="/global/cfs/cdirs/lsst/groups/CS/descqa/lib/kcorrect"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$KCORRECT_DIR/lib"

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v1"
URL="https://portal.nersc.gov/projecta/lsst/descqa/v1/"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import descqarun; descqarun.main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" -p v1 -w "$URL" "$@"

# end subshell
)
