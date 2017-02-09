#!/bin/bash

# go to a subshell
(

# make sure all commands are executed
set -e

# set DESCQA root directory
DESCQAROOTDIR="/scratch2/scratchdirs/yymao/descqa-backup"

# load python module and set PYTHONPATH
export PYTHONPATH=""
module load python/2.7.9
module load numpy/1.9.2
module load scipy/0.15.1
module load matplotlib/1.4.3
module load h5py/2.5.0
export PYTHONPATH="$DESCQAROOTDIR/lib/python2.7/site-packages:$PYTHONPATH"

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
