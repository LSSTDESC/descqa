#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=32

export OMP_NUM_THREADS=32

./run_master.sh -p /global/cfs/projectdirs/lsst/groups/SRV/gcr-catalogs -c skysim5000_v1.2 -t tpcf_Wang2013_rSDSS_jack
