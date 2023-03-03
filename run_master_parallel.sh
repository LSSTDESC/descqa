#!/bin/bash

#HOSTS=.hosts-job$SLURM_JOB_ID
#HOSTFILE=.hostlist-job$SLURM_JOB_ID
#srun hostname -f > $HOSTS
#sort $HOSTS | uniq -c | awk '{print $2 ":" $1}' >> $HOSTFILE
echo "--------- Running SRUN script ---------"

# go to a subshell
(

# make sure all commands are executed
#set -e

# activate DESC python environment
source /global/common/software/lsst/common/miniconda/setup_current_python.sh ""
PYTHON='python'

# set output directory
OUTPUTDIR="/global/cfs/cdirs/lsst/groups/CS/descqa/run/v2"

# to allow wildcards in arguments go to master.py
set -o noglob

# run master.py
CMD="import mpi4py; import descqarun; descqarun.main()"
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8

srun -n 2 $PYTHON -E -c "$CMD" "$OUTPUTDIR" "$@"

# end subshell
)

#rm $HOSTS $HOSTFILE
