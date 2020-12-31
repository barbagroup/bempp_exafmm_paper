#!/bin/bash
#SBATCH --job-name=6CO8_juffer_q2_p4
#SBATCH --output=6CO8_q2_p4_%j.out
#SBATCH --error=6CO8_q2_p4_%j.err
#SBATCH --partition defq

module load openblas
module load intel/parallel_studio_xe

/usr/bin/time -v python ${REPRO_PATH}/scripts/juffer_ex.py ${REPRO_PATH}/runs/6CO8/config/6CO8_q2_p4.yml
