#!/bin/bash

module load python/2.7.9
module load numpy/1.9.2
module load scipy/0.15.1
module load matplotlib/1.4.3
module load h5py/2.5.0
module load cython/0.22

DESCQAROOTDIR="/project/projectdirs/lsst/descqa"

export PYTHONPATH="$DESCQAROOTDIR/lib/python2.7/site-packages:$PYTHONPATH"
OUTPUTDIR="$DESCQAROOTDIR/run/edison"
CATALOGDIR="$DESCQAROOTDIR/catalog"

umask 0002 # to ensure permission of output is correct
python master.py $OUTPUTDIR --catalog-dir $CATALOGDIR $@
