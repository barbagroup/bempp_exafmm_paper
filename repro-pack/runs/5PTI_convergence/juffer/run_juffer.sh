#!/bin/bash
#SBATCH --job-name=5PTI_convergence_juffer
#SBATCH --output=5PTI_%A_%a.out
#SBATCH --error=5PTI_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition defq
#SBATCH --array=0,1,2,3,4

module load openblas
module load intel/parallel_studio_xe

DENSITY_ARR=(1 2 4 8 16)
DENSITY=${DENSITY_ARR[${SLURM_ARRAY_TASK_ID}]}

mkdir -p $DENSITY
cd $DENSITY
/usr/bin/time -v python ${REPRO_PATH}/scripts/juffer.py ${REPRO_PATH}/runs/5PTI_convergence/config/5PTI_ed${DENSITY}.yml
cd ..
rm -rf $DENSITY
