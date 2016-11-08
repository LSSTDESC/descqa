#!/bin/bash

module load python
module load numpy
module load matplotlib
module load h5py

export PYTHONPATH="/project/projectdirs/lsst/descqa/src/flashTest/lib/python2.7/site-packages:$PYTHONPATH"
umask 0002
OUTPUTDIR="/project/projectdirs/lsst/descqacmu/run/edison"
python master.py $OUTPUTDIR $@
