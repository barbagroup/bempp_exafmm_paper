#!/bin/bash
#SBATCH --job-name=sphere_convergence_derivative
#SBATCH --output=sphere_%A_%a.out
#SBATCH --error=sphere_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition defq
#SBATCH --array=0,1,2,3,4

module load openblas
module load intel/parallel_studio_xe

SIZE_ARR=(512 2048 8192 32768 131072)
SIZE=${SIZE_ARR[${SLURM_ARRAY_TASK_ID}]}

mkdir -p $SIZE
cd $SIZE
/usr/bin/time -v python ${REPRO_PATH}/bempp_pbs/scripts/derivative_ex_mass_matrix.py ${REPRO_PATH}/runs/sphere_convergence/config/sphere${SIZE}_r4.yml
cd ..
rm -rf $SIZE
