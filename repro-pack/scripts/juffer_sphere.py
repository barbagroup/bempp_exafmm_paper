# Usage: python juffer_sphere.py [config.yml]
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
dirichl_space = bempp.api.function_space(grid, "P", 1)
neumann_space = dirichl_space

# define operators
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz

phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
ep = ep_ex/ep_in

dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=assembler)
dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
L1 = (ep*dP) - dF

F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=assembler)
P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
L2 = F - P

ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=assembler)
ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler=assembler)
L3 = ddP - ddF

dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=assembler)
dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler=assembler)
L4 = dF0 - ((1.0/ep)*dP0)

A = bempp.api.BlockedOperator(2, 2)
A[0, 0] = (0.5*(1.0 + ep)*phi_id) - L1
A[0, 1] = (-1.0)*L2
A[1, 0] = L3
A[1, 1] = (0.5*(1.0 + (1.0/ep))*dph_id) - L4

# assembly the system matrix (blocked_operator)
start = time.time()
A_op = A.strong_form()
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

@bempp.api.callable(vectorized=True)
def fmm_d_green_func(x, n, domain_index, result):
    import exafmm.laplace as _laplace
    sources = _laplace.init_sources(x_q, q)
    targets = _laplace.init_targets(x.T)
    fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
    tree = _laplace.setup(sources, targets, fmm)
    values = _laplace.evaluate(tree, fmm)
    os.remove('.rhs.tmp')
    result[:] = np.sum(values[:,1:] * n.T, axis=1) / ep_in

rhs1 = bempp.api.GridFunction(dirichl_space, fun=fmm_green_func)
rhs2 = bempp.api.GridFunction(dirichl_space, fun=fmm_d_green_func)
stop = time.time()
print("generate rhs GridFunction wall time: {}".format(stop-start))

# generate rhs vector as numpy array
from bempp.api.assembly.blocked_operator import (
    coefficients_from_grid_functions_list,
    grid_function_list_from_coefficients)
rhs_vec = coefficients_from_grid_functions_list([rhs1, rhs2])

# solve the linear system
from scipy.sparse.linalg import gmres
from bempp.api.linalg.iterative_solvers import IterationCounter
callback = IterationCounter(True)

start = time.time()
sol_vec, info = gmres(A_op, rhs_vec, tol=tol, restart=restart, callback=callback)
stop = time.time()

print("gmres wall time: {}".format(stop-start))
print("The linear system was solved in {0} iterations".format(callback.count))

sol = grid_function_list_from_coefficients(sol_vec.ravel(), A.domain_spaces)
solution_dirichl, solution_neumann = sol

# save result into npz
np.savez('solution', solution_dirichl=solution_dirichl.coefficients, solution_neumann=solution_neumann.coefficients)
slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose(), assembler='fmm')
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose(), assembler='fmm')
phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

# total dissolution energy applying constant to get units [kcal/mol]
total_energy = 2 * np.pi * 332.064 * np.sum(q * phi_q).real
print("Total solvation energy: {:.8f} [kcal/Mol]".format(total_energy))