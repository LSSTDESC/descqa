#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate python env
PYTHON="/global/common/cori/contrib/lsst/apps/anaconda/py3-envs/DESCQA/bin/python"

# set output directory
OUTPUTDIR="/global/projecta/projectdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="from descqarun.master import main; main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"

# end subshell
)
