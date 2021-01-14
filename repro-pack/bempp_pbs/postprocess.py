"""
Utilities for post-processing including parsing results and plotting.
"""

# preferred plot style for publication
PLOT_PARAMS = {'font.family': 'serif',
			   'legend.fontsize': 6,
			   'legend.handlelength': 2,
			   'figure.dpi': 300,
			   'lines.markersize': 4,
			   'lines.markeredgewidth': 0.5,
			   'lines.linewidth': 0.4,
			   'axes.titlesize': 8,
			   'axes.labelsize': 8,
			   'xtick.labelsize': 8,
			   'ytick.labelsize': 8,
			   'xtick.major.width': 0.6,
			   'xtick.minor.width': 0.4,
			   'ytick.major.width': 0.6,
			   'ytick.minor.width': 0.4,
			   'xtick.direction': 'in',
			   'ytick.direction': 'in',
			   'grid.linewidth': 0.2}


def _bempp_parser(res_file, formulation='direct', skip4=False, debug=False):
    """
    A helper to that parse the log from Bempp-cl.
    """
    import numpy as np
    result = dict()
    t_fmm_eval = list()
    t_fmm_init = list()
    t_assemble_sparse = list()
    t_singular_assembler = list()
    t_singular_correction = list()
    t_solver_init = list()
    t_solver_solve = list()
    if formulation == 'juffer':
        num_laplace_fmm = 8
        num_helmholtz_fmm = 11
    elif formulation == 'direct':   
        num_laplace_fmm = 4
        num_helmholtz_fmm = 4
    else:
        print('formulation is not correct!')
        return
    with open(res_file) as f:
        lines = f.readlines()
        for line in lines:
            if 'bempp:HOST:TIMING: Finished Operation: Calling ExaFMM.:' in line:          
                t_fmm_eval.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: Finished Operation: Initialising Exafmm.:' in line:
                t_fmm_init.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: Finished Operation: Singular assembler:' in line:
                t_singular_assembler.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: assemble_sparse :' in line:
                t_assemble_sparse.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: _Solver.__init__ :' in line:
                t_solver_init.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: _Solver.solve :' in line:
                t_solver_solve.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'bempp:HOST:TIMING: Finished Operation: Singular Corrections Evaluator: ' in line:
                t_singular_correction.append(float(line.strip().split(':')[-1].strip('s')))
            elif 'The linear system was solved in' in line:
                num_iter = int(line.strip().split()[-2])
                result['num_iter'] = num_iter
            elif 'gmres wall time' in line:
                t_gmres_wall = float(line.strip().split()[-1])
                result['t_total_gmres'] = t_gmres_wall
            elif 'number of elements' in line:
                num_elem = int(line.strip().split()[-1])
                result['num_elem'] = num_elem
            elif 'Total solvation energy' in line:
                e_solv = float(line.strip().split()[-2])
                result['e_solv [kcal/Mol]'] = e_solv
            elif 'assembly system matrix wall time:' in line:
                t_total_assembly = float(line.strip().split()[-1])
                result['t_total_assembly'] = t_total_assembly
            elif 'fmm ncrit' in line:
                ncrit = int(line.strip().split()[-1])
                result['ncrit'] = ncrit
            elif 'fmm tree depth' in line:
                depth = int(line.strip().split()[-1])
                result['depth'] = depth
            elif 'Maximum resident set size' in line:
                memory = int(line.strip().split()[-1]) / 1e6   # convert kb to gb
                result['memory [GB]'] = memory
            elif 'fmm expansion order' in line:
                exp_order = int(line.strip().split()[-1])
                result['exp_order'] = exp_order
    
    # skip last 4 fmm timings (used in computing off-boundary potentials)
    if skip4:
        t_fmm_eval = t_fmm_eval[:-4]
        t_fmm_init = t_fmm_init[:-1]
    t_fmm_eval = np.array(t_fmm_eval).reshape(-1,num_laplace_fmm+num_helmholtz_fmm)
    t_fmm_eval.sort(axis=1)  # sort each row
    t_laplace = t_fmm_eval[:, :num_laplace_fmm]
    t_helmholtz = t_fmm_eval[:, num_laplace_fmm:]
    # print(t_laplace.shape, t_helmholtz.shape)
    t_avg_laplace = t_laplace.mean()
    t_avg_helmholtz = t_helmholtz.mean()
    
    
    if debug:
        print('t_laplace.shape', t_laplace.shape)
        print('t_helmholtz.shape', t_helmholtz.shape)
        print('t_singular_assembler', len(t_singular_assembler))
        print('t_singular_correction', len(t_singular_correction))
        print('t_assemble_sparse', len(t_assemble_sparse))
        print('t_solver_solve', len(t_solver_solve))
    
    result['t_fmm_init'] = np.array(t_fmm_init).sum()
    #result['t_solver_init'] = np.array(t_solver_init).sum()
    #result['t_solver_solve'] = np.array(t_solver_solve).sum()
    result['t_singular_assembler'] = np.array(t_singular_assembler).sum()
    result['t_assemble_sparse'] = np.array(t_assemble_sparse).sum()
    result['t_assembly_other'] = result['t_total_assembly'] - result['t_fmm_init'] - result['t_singular_assembler'] - result['t_assemble_sparse']
    
    result['t_singular_correction'] = np.array(t_singular_correction).sum()
    result['t_laplace'] = t_laplace.sum()
    result['t_helmholtz'] = t_helmholtz.sum()
    result['t_avg_laplace'] = t_avg_laplace
    result['t_avg_helmholtz'] = t_avg_helmholtz
    result['t_gmres_other'] = result['t_total_gmres'] - result['t_laplace'] - result['t_helmholtz'] - result['t_singular_correction']

    return result

def get_df(directory, formulation='direct', skip4=False):
    """
    Gather raw result files from a directory to produce a pandas DataFrame.
    """
    from os.path import join, isfile
    import os
    import glob
    import pandas as pd
    
    tmp_results = list()
    
    for res_file in glob.glob(join(directory, '*.out')):
        err_file = res_file.replace('.out', '.err')
        # append .err to .out (if any)
        if isfile(err_file):
            with open(res_file, 'ab') as outfile:
                with open(err_file, "rb") as infile:
                    outfile.write(infile.read())
        # remove .err file
        try:
            os.remove(err_file)
        except OSError:
            pass
                    
        tmp_results.append(_bempp_parser(res_file, formulation, skip4))
    
    results_dict = dict()
    for key in tmp_results[0].keys():
        results_dict[key] = tuple(result[key] for result in tmp_results)
    return pd.DataFrame(results_dict).set_index('num_elem').sort_index()
