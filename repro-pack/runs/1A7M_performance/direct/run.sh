#!/bin/bash
#SBATCH --job-name=1A7M_performance_direct
#SBATCH --output=1A7M_%A_%a.out
#SBATCH --error=1A7M_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition nano
#SBATCH --array=0,1,2,3,4
#SBATCH --time=00:30:00

module load openblas
module load intel/parallel_studio_xe

REFINE_ARR=(1 2 3 4 5)
REFINE=${REFINE_ARR[${SLURM_ARRAY_TASK_ID}]}

mkdir -p $REFINE
cd $REFINE
/usr/bin/time -v python ${REPRO_PATH}/bempp_pbs/scripts/direct.py ${REPRO_PATH}/runs/1A7M_performance/config/1A7M_${REFINE}.yml
cd ..
rm -rf $REFINE
