#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# set DESCQA root directory
DESCQAROOTDIR="/global/projecta/projectdirs/lsst/groups/CS/descqa"

# activate python env
export PYTHONPATH=""
source /global/common/cori/contrib/lsst/apps/anaconda/4.4.0-py3/bin/activate ""
source activate DESCQA

# for kcorrect 
export KCORRECT_DIR="$DESCQAROOTDIR/lib/kcorrect"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$KCORRECT_DIR/lib"

# set other necessary paths
OUTPUTDIR="$DESCQAROOTDIR/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
python master.py "$OUTPUTDIR" "$@"

# end subshell
)

