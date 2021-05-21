#!/bin/bash
#SBATCH --job-name=1A2S_convergence_derivative
#SBATCH --output=1A2S_%A_%a.out
#SBATCH --error=1A2S_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition debug-cpu
#SBATCH --array=0,1,2
#SBATCH --time=02:00:00               # Time limit hrs:min:sec

module load openblas
module load intel/parallel_studio_xe

DENSITY_ARR=(4 8 16)
DENSITY=${DENSITY_ARR[${SLURM_ARRAY_TASK_ID}]}

mkdir -p $DENSITY
cd $DENSITY
/usr/bin/time -v python ${REPRO_PATH}/bempp_pbs/scripts/derivative_ex.py ${REPRO_PATH}/runs/1A2S_convergence/config/1A2S_ed${DENSITY}.yml
cd ..
rm -rf $DENSITY
