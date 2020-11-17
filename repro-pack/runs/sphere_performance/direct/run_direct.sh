#!/bin/bash
#SBATCH --job-name=sphere_performance_direct
#SBATCH --output=sphere_%A_%a.out
#SBATCH --error=sphere_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition defq
#SBATCH --array=0,1,2,3,4

module load openblas
module load intel/parallel_studio_xe

REFINE_ARR=(5 6 7 8 9)
REFINE=${REFINE_ARR[${SLURM_ARRAY_TASK_ID}]}

mkdir -p $REFINE
cd $REFINE
/usr/bin/time -v python ${REPRO_PATH}/scripts/direct_sphere.py ${REPRO_PATH}/runs/sphere_performance/config/sphere_refine${REFINE}.yml
cd ..
rm -rf $REFINE
