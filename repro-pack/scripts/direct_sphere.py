# Usage: python direct_sphere.py [config.yml]
import bempp.api
import numpy as np
import sys
import os
import time
import yaml
from helpers import generate_grid, parse_pqr

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
mesh_dir = os.path.join(parent_dir, "mesh")
pqr_dir = os.path.join(parent_dir, "pqr")

with open(sys.argv[1]) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    ep_in = data['ep_in']
    ep_ex = data['ep_ex']
    kappa = data['kappa']
    bempp.api.GLOBAL_PARAMETERS.quadrature.regular = data['regular']
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = data['expansion_order']
    bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = data['ncrit']
    restart = data['restart']
    tol = data['tol']
    refine_level = int(data['refine_level'])

# create grid and some random charges
grid = bempp.api.shapes.regular_sphere(refine_level)
n_q = 100
rand = np.random.RandomState(0)
q = rand.rand(n_q)  # charges
x_q = rand.rand(3*n_q) - 0.5
x_q = x_q.reshape((n_q,3))
x_q = 0.7 * x_q / np.linalg.norm(x_q, axis=1).reshape((n_q,1))

print("number of elements: {}".format(grid.number_of_elements))
bempp.api.enable_console_logging('debug')
assembler = "fmm"

# define function space
dirichl_space = bempp.api.function_space(grid, "DP", 0)
neumann_space = dirichl_space

# define operators
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=assembler)
dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=assembler)
slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=assembler)

A = bempp.api.BlockedOperator(2, 2)
A[0, 0] = 0.5*identity + dlp_in
A[0, 1] = -slp_in
A[1, 0] = 0.5*identity - dlp_out
A[1, 1] = (ep_in/ep_ex)*slp_out

# assembly the system matrix (blocked_operator)
start = time.time()
A_weak_form = A.weak_form()

# block-diagonal preconditioner
identity_diag = identity.weak_form().A.diagonal()
slp_in_diag = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
dlp_in_diag = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler="only_diagonal_part").weak_form().A
slp_out_diag = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A
dlp_out_diag = modified_helmholtz.double_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler="only_diagonal_part").weak_form().A

diag11 = .5 * identity_diag + dlp_in_diag
diag12 = -slp_in_diag
diag21 = .5 * identity_diag - dlp_out_diag
diag22 = (ep_in / ep_ex) * slp_out_diag

d_aux = 1 / (diag22 - diag21 * diag12 / diag11)
diag11_inv = 1/diag11 + 1/diag11 * diag12 * d_aux * diag21 / diag11
diag12_inv = -1/diag11 * diag12 * d_aux
diag21_inv = -d_aux * diag21 / diag11
diag22_inv = d_aux

from scipy.sparse import diags, bmat
from scipy.sparse.linalg import aslinearoperator
block_mat_precond = bmat([[diags(diag11_inv), diags(diag12_inv)],
                          [diags(diag21_inv), diags(diag22_inv)]]).tocsr()
precond = aslinearoperator(block_mat_precond)

stop = time.time()
print("assembly system matrix wall time: {}".format(stop-start))

# generate rhs GridFunction
start = time.time()

@bempp.api.callable(vectorized=True)
def fmm_green_func(x, n, domain_index, result):
    import exafmm.laplace as _laplace
    sources = _laplace.init_sources(x_q, q)
    targets = _laplace.init_targets(x.T)
    fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
    tree = _laplace.setup(sources, targets, fmm)
    values = _laplace.evaluate(tree, fmm)
    os.remove('.rhs.tmp')
    result[:] = values[:,0] / ep_in

@bempp.api.real_callable
def zero_func(x, n, domain_index, result):
    result[0] = 0

rhs1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)
rhs2 = bempp.api.GridFunction(neumann_space, fun=zero_func)

stop = time.time()
print("generate rhs GridFunction wall time: {}".format(stop-start))

# generate rhs vector as numpy array
from bempp.api.assembly.blocked_operator import (
    projections_from_grid_functions_list,
    grid_function_list_from_coefficients)
rhs_vec = projections_from_grid_functions_list([rhs1, rhs2], A.dual_to_range_spaces)

# solve the linear system
from scipy.sparse.linalg import gmres
from bempp.api.linalg.iterative_solvers import IterationCounter
callback = IterationCounter(True)

start = time.time()
sol_vec, info = gmres(A_weak_form, rhs_vec, M=precond, callback=callback, tol=tol, restart=restart)
stop = time.time()

np.save('solution', sol_vec)
print("gmres wall time: {}".format(stop-start))
print("The linear system was solved in {0} iterations".format(callback.count))

sol = grid_function_list_from_coefficients(sol_vec.ravel(), A.domain_spaces) # convert solution vector to grid_function
solution_dirichl, solution_neumann = sol
slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose(), assembler='fmm')
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose(), assembler='fmm')
phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

# total dissolution energy applying constant to get units [kcal/mol]
total_energy = 2 * np.pi * 332.064 * np.sum(q * phi_q).real
print("Total solvation energy: {:.8f} [kcal/Mol]".format(total_energy))
