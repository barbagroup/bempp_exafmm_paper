# Reproducibility Package

This is the repro-pack for the bimolecular electrostatics applications using bempp-cl and exafmm-t. It has the following structure:

- `bempp_pbs`: a Python package thats uses bempp-cl to solve Poisson-Boltzmann problems.
- `mesh`: stores mesh files. We use MSMS mesh format and hence each mesh consists of two files: `[mesh_name].vert` for vertex information and `[mesh_name].face` for facet information.
- `pqr`: stores `.pqr` files.
- `runs`: this is where we run the experiments. Each subdirectory is dedicated to a specific study, and contains a slurm job submission script and configuration files for that study.
- `notebooks`: contains Jupyter notebooks that post-process the raw data from `runs` folder and generate all results that are presented in the manuscript.

Following the steps below to run each case:

**1. Install bempp-cl and exafmm-t**

We suggest to use `conda` as the package manager and create a conda environment for this application.
``` bash
conda create --yes -n bempp python=3.8
conda install -n bempp --yes numpy scipy matplotlib numba pytest jupyter plotly git pip mpi4py pyyaml
conda install -n bempp --yes -c conda-forge pocl pyopencl meshio
```
Then activate this environment: `conda activate bempp`.
Finally, install `bempp-cl` and `exafmm-t`:
``` bash
pip install git+git://github.com/bempp/bempp-cl
pip install git+git://github.com/exafmm/exafmm-t
```

The installation will take several minutes on a normal workstation.

**2. Clone this repo and install bempp_pbs**

Next, clone this repo locally, change directory to this `repro-pack` folder:
``` bash
git clone https://github.com/barbagroup/bempp_exafmm_paper.git
cd bempp_exafmm_paper/repro-pack 
```
Then pip install `bempp_pbs` in editable mode:
``` bash
pip install -e .
```

**3. Run/Sumbit the script**
First, the current scripts need us to define the environment variable `REPRO_PATH` as this `bempp_exafmm_paper/repro-pack` directory:
``` bash
export REPRO_PATH=$(pwd)
```

The structure of each folder in `runs` looks like this:
```
runs/6CO8
├── config
│   └── 6CO8_q2_p4.yml
├── derivative_ex
│   ├── 6CO8_q2_p4_920333.out
│   └── run.sh
└── direct
    ├── 6CO8_q2_p4_897379.out
    └── run.sh
```
For example, to sumbit the run for the Zika virus case using derivative formulation, you simply need:
``` bash
cd runs/6CO8/derivative_ex
sbatch run.sh
```

The raw output file, in this case `6CO8_q2_p4_920333.out`, will be saved in the same directory. Next, you can follow [the corresponding notebook](https://github.com/barbagroup/bempp_exafmm_paper/blob/master/repro-pack/notebooks/6CO8.ipynb) to post-process the data.

For users whose environment does not use slurm workload manager, you can extract the execution command from `run.sh`. To run the same case as above:
``` bash
python ${REPRO_PATH}/bempp_pbs/scripts/derivative_ex.py ${REPRO_PATH}/runs/6CO8/config/6CO8_q2_p4.yml
```
