#!/bin/bash
#SBATCH -A m1727
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128

export OMP_NUM_THREADS=32

./run_master.sh -c skysim5000_v1.2 -t shear
