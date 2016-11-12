#!/bin/bash

# set DESCQA root directory
DESCQAROOTDIR="/project/projectdirs/lsst/descqa"

# load python module and set PYTHONPATH
export PYTHONPATH=""
module load python/2.7.9
module load numpy/1.9.2
module load scipy/0.15.1
module load matplotlib/1.4.3
module load h5py/2.5.0
module load cython/0.22
export PYTHONPATH="$DESCQAROOTDIR/lib/python2.7/site-packages:$PYTHONPATH"

# set other necessary paths
OUTPUTDIR="$DESCQAROOTDIR/run/edison"
CATALOGDIR="$DESCQAROOTDIR/catalog"

# ensure permission of output is readable by the web interface
umask 0002

# run master.py
python master.py $OUTPUTDIR --catalog-dir $CATALOGDIR $@

