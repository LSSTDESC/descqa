#!/bin/bash
# go to a subshell
(

# make sure all commands are executed
#set -e

# activate DESC python environment
#source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""
#source /cvmfs/sw.lsst.eu/linux-x86_64/lsst_distrib/w_2022_41/loadLSST.bash "" 
#source /global/common/software/lsst/common/miniconda/kernels/stack-weekly.sh ""
PYTHON='python'
#echo $PYTHONPATH > a.out
source /global/homes/p/plarsen/plarsen_git/lsst_stack/loadLSST.bash
setup lsst_distrib
#echo $PYTHONPATH > b.out
# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
OMP_NUM_THREADS=8
#CMD="/global/common/software/lsst/common/miniconda/start-kernel-cli.py desc-stack-weekly"
#CMD2="import sys; sys.path.insert(0,'/global/homes/p/plarsen/plarsen_git/descqa'); sys.path.insert(0,'/global/homes/p/plarsen/plarsen_git/gcr-catalogs'); sys.path.insert(0,'/global/homes/p/plarsen/plarsen_git/generic-catalog-reader'); sys.path.insert(0,'/global/homes/p/plarsen/plarsen_git/easyquery'); import GCR; import GCRCatalogs; import descqarun; descqarun.main()"
#mpirun -n 1 
CMD="import descqarun; descqarun.main()"
mpirun -n 1 /global/homes/p/plarsen/plarsen_git/lsst_stack/conda/miniconda3-py38_4.9.2/envs/lsst-scipipe-4.0.0/bin/python -c "$CMD" "$OUTPUTDIR" "$@"
#export OMP_NUM_THREADS=8

# end subshell
)
