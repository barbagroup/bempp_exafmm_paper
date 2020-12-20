# Usage: python juffer_in.py [config.yml]
import bempp.api
import numpy as np
import sys
import time
import helpers
import formulations

bempp.api.enable_console_logging('debug')
assembler = "fmm"

# parse configurations and generate grid
grid, q, x_q = helpers.parse_config(sys.argv[1])
bempp.api.GLOBAL_PARAMETERS.quadrature.regular = helpers.params['regular']
bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = helpers.params['expansion_order']
bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = helpers.params['ncrit']
print(f"number of elements: {grid.number_of_elements}")

# define function space
dirichl_space = bempp.api.function_space(grid, "P", 1)
neumann_space = dirichl_space

# define system matrix and RHS
A, rhs1, rhs2 = formulations.juffer_in(dirichl_space, neumann_space, assembler, q, x_q)

# assembly system matrix (including preparing preconditioner)
start = time.time()
A_op = A.strong_form()
stop = time.time()
print(f"assembly system matrix wall time: {stop-start}")

# generate RHS vector as numpy array
from bempp.api.assembly.blocked_operator import coefficients_from_grid_functions_list
rhs_vec = coefficients_from_grid_functions_list([rhs1, rhs2])

# solve the linear system
from scipy.sparse.linalg import gmres
from bempp.api.linalg.iterative_solvers import IterationCounter
callback = IterationCounter(True)

start = time.time()
sol_vec, info = gmres(A_op, rhs_vec, callback=callback, tol=helpers.params['tol'], restart=helpers.params['restart'])
stop = time.time()

if 'save_solution' in helpers.params.keys():
    if helpers.params['save_solution']:
        np.save('solution', sol_vec)
print(f"gmres wall time: {stop-start}")
print(f"The linear system was solved in {callback.count} iterations")

# compute solvation energy
from bempp.api.assembly.blocked_operator import grid_function_list_from_coefficients
sol = grid_function_list_from_coefficients(sol_vec.ravel(), A.domain_spaces)
solution_dirichl, solution_neumann = sol
slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose(), assembler='fmm')
dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose(), assembler='fmm')
phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

# total dissolution energy applying constant to get units [kcal/mol]
total_energy = 2 * np.pi * 332.064 * np.sum(q * phi_q).real
print(f"Total solvation energy: {total_energy:.8f} [kcal/Mol]")
