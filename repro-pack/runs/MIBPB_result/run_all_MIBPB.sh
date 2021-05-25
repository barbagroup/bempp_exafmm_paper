#!/bin/bash
#SBATCH --job-name=mibpb_all
#SBATCH --output=mibpb_%j.out
#SBATCH --error=mibpb_%j.err
#SBATCH --partition defq
#SBATCH --time=10:00:00               # Time limit hrs:min:sec

#!/bin/bash

h_arr=(1.00 0.50 0.25)
id_arr=(1AJJ 1VJW 1A2S 1A7M 1F6W 1A63 1R69 1SVR 5PTI)

for id in ${id_arr[@]}
do
  mkdir -p $id
  for h in ${h_arr[@]}
  do
    output="${id}/${id}_h${h}.dat"
    /usr/bin/time -v ./mibpb $id eps0=4 eps1=80 kappa1=0.125 h=$h prds=1.4 ilinear=1 smibpb=0 &> $output
  done
done
