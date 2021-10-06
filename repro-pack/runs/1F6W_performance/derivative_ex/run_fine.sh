#!/bin/bash
#SBATCH --job-name=1F6W
#SBATCH --output=1F6W_fine_%j.out
#SBATCH --error=1F6W_fine_%j.err
#SBATCH --partition debug-cpu
#SBATCH --time=04:00:00

module load openblas
module load intel/parallel_studio_xe

/usr/bin/time -v python ${REPRO_PATH}/bempp_pbs/scripts/derivative_ex.py ${REPRO_PATH}/runs/1F6W_performance/config/1F6W_fine.yml
