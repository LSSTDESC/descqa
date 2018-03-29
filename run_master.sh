#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# load needed modules
module load gsl/2.1
module load cray-fftw/3.3.6.2

# activate python env
PYTHON="/global/common/software/lsst/common/miniconda/py3-4.2.12/bin/python"

# set output directory
OUTPUTDIR="/global/projecta/projectdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import descqarun; descqarun.main()"
$PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"

# end subshell
)
