#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# set DESCQA root directory
DESCQAROOTDIR="/global/projecta/projectdirs/lsst/descqa"

# load python module and set PYTHONPATH
module unload python
export PYTHONPATH=""
module load python/2.7.9
module load h5py/2.5.0
export PYTHONPATH="$DESCQAROOTDIR/lib/python2.7/site-packages:$PYTHONPATH"

# for kcorrect 
export KCORRECT_DIR="$DESCQAROOTDIR/lib/kcorrect"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$KCORRECT_DIR/lib"

# set other necessary paths
OUTPUTDIR="$DESCQAROOTDIR/run/edison"
CATALOGDIR="$DESCQAROOTDIR/catalog"

# ensure permission of output is readable by the web interface
umask 0002

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
python master.py "$OUTPUTDIR" --catalog-dir "$CATALOGDIR" "$@"

# end subshell
)
