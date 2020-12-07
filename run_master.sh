#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate DESC python environment
source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import descqarun; descqarun.main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"

# end subshell
)
