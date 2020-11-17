# Usage: mpirun -np [n] python juffer_mpi.py [config.yml]
import bempp.api
from bempp.api.utils import remote_operator
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
    bempp.api.GLOBAL_PARAMETERS.fmm.expansion_order = data['expansion_order']
    bempp.api.GLOBAL_PARAMETERS.fmm.ncrit = data['ncrit']
    tol = data['tol']

    face_file = os.path.join(mesh_dir, data['face_file'])
    vert_file = os.path.join(mesh_dir, data['vert_file'])
    pqr_file  = os.path.join(pqr_dir, data['pqr_file'])
    grid = generate_grid(face_file, vert_file)
    q, x_q = parse_pqr(pqr_file)

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

op00 = (0.5*(1.0 + ep)*phi_id) - L1
op01 = (-1.0)*L2
op10 = L3
op11 = (0.5*(1.0 + (1.0/ep))*dph_id) - L4

# Initialise the parallel manager
manager = remote_operator.get_remote_manager()

# We turn op into remote operators by registering with the manager
manager.register(op00)
manager.register(op01)
manager.register(op10)
manager.register(op11)

# Initialise remote operators
A = remote_operator.RemoteBlockedOperator(2, 2)
A[0, 0] = op00
A[0, 1] = op01
A[1, 0] = op10
A[1, 1] = op11


if remote_operator.MPI_RANK != 0:
    # On ranks that are not zero we start the worker process
    # This takes compute tasks, executes them, and sends results back to rank 0
    manager.execute_worker()
else:
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
    print("generate rhs wall time: {}".format(stop-start))

    # solve the linear system
    start = time.time()
    sol, info, it_count = bempp.api.linalg.gmres(A, [rhs1, rhs2],
                                                use_strong_form=True, return_iteration_count=True, tol=tol)
    stop = time.time()
    print("gmres wall time: {}".format(stop-start))
    print("The linear system was solved in {0} iterations".format(it_count))

    solution_dirichl, solution_neumann = sol
    slp_q = bempp.api.operators.potential.laplace.single_layer(neumann_space, x_q.transpose(), assembler='fmm')
    dlp_q = bempp.api.operators.potential.laplace.double_layer(dirichl_space, x_q.transpose(), assembler='fmm')
    phi_q = slp_q * solution_neumann - dlp_q * solution_dirichl

    # total dissolution energy applying constant to get units [kcal/mol]
    total_energy = 2 * np.pi * 332.064 * np.sum(q * phi_q).real
    print("Total solvation energy: {:7.2f} [kcal/Mol]".format(total_energy))

    manager.shutdown()
