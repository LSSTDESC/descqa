#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
#set -e

# activate DESC python environment
source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""
PYTHON='python'

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
OMP_NUM_THREADS=1
CMD="import descqarun; descqarun.main()"
mpirun -n 1 $PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"
#export OMP_NUM_THREADS=8

# end subshell
)
