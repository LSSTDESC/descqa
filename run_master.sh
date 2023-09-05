#!/bin/bash

echo "--------- Running SRUN script ---------"

# go to a subshell
(

# make sure all commands are executed
#set -e

############# options  #############
#
# default is cpu perlmutter setup, in parallel on 32 ranks


# default to cpu if you're using CPU nodes, otherwise set this to false for GPU setup
cpu=true

# whether you're running the analysis tools test, which requires the stack
analysistools=false

# if you want to run the parallel version
parallel=true

# number of threads and ranks to run on
OMP_NUM_THREADS=4
NUMEXPR_MAX_THREADS=4
if parallel
then 
RANKS=32
else
RANKS=1
fi 

#################################

# activate DESC python environment
if analysistools 
then 
source /global/homes/p/plarsen/plarsen_git/lsst_stack/loadLSST.bash
setup lsst_distrib
elif $cpu
then 
source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""
else
source /global/common/software/lsst/common/miniconda/setup_dev_python.sh ""
#source /global/common/software/lsst/common/miniconda/setup_gpu_python.sh long term will likely change to this, check with Heather for current usage 
fi

if analysistools
then
PYTHON="/global/homes/p/plarsen/plarsen_git/lsst_stack/conda/miniconda3-py38_4.9.2/envs/lsst-scipipe-4.0.0/bin/python"
else
PYTHON='python'
fi 

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

if parallel
then
CMD="import mpi4py; import descqarun; descqarun.main()"
else
CMD="import descqarun; descqarun.main()"
fi

export OMP_NUM_THREADS
export NUMEXPR_MAX_THREADS

srun -n $RANKS $PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"
	

# end subshell
)

