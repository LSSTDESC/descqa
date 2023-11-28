#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# activate DESC python environment
source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""
PYTHON='python'

# increase maximum number of threads (default is 8)
export NUMEXPR_MAX_THREADS=256
# set number of threads to prevent thread error
export OMP_NUM_THREADS=64

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import descqarun; descqarun.main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"

# end subshell
)
