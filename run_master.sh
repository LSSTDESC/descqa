#!/bin/bash

module load python
module load numpy
module load matplotlib
module load h5py

DESCQAROOTDIR="/project/projectdirs/lsst/descqacmu"

export PYTHONPATH="$DESCQAROOTDIR/lib/python2.7/site-packages:$PYTHONPATH"
OUTPUTDIR="$DESCQAROOTDIR/run/edison"
CATALOGDIR="$DESCQAROOTDIR/catalog"

umask 0002 # to ensure permission of output is correct
python master.py $OUTPUTDIR --catalog-dir $CATALOGDIR $@
