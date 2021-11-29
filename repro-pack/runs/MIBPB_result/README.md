### MIBPB results

This folder contains the input files, running scripts and raw output for the MIBPB results reported in the first revision.
The MIBPB binary we used was downloaded on May 10th, 2021 from the [MIBPB website](https://weilab.math.msu.edu/MIBPB/), and there was no mentioning of its software version at the time.
The bash script `run_all_MIBPB.sh` will automatically run all cases for 9 proteins.
In the script, you will find the parameters that we used: `eps0=4 eps1=80 kappa1=0.125 prds=1.4 ilinear=1 smibpb=0`.
For each protein, we used three grid spacings `h=1.00`, `h=0.50` and `h=0.25` for the convergence study.
The raw outputs for each protein sit in a separate folder, named as its PDB ID.
