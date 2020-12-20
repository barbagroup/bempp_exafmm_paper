import helpers
import bempp.api
import numpy as np
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
import os

def direct(dirichl_space, neumann_space, assembler, q, x_q):
    ep_in = helpers.params['ep_in']
    ep_ex = helpers.params['ep_ex']
    kappa = helpers.params['kappa']

    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    slp_in   = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=assembler)
    dlp_in   = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=assembler)
    slp_out  = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
    dlp_out  = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=assembler)

    A_sys = bempp.api.BlockedOperator(2, 2)
    A_sys[0, 0] = 0.5*identity + dlp_in
    A_sys[0, 1] = -slp_in
    A_sys[1, 0] = 0.5*identity - dlp_out
    A_sys[1, 1] = (ep_in/ep_ex)*slp_out

    @bempp.api.callable(vectorized=True)
    def rhs1_fun(x, n, domain_index, result):
        import exafmm.laplace as _laplace
        sources = _laplace.init_sources(x_q, q)
        targets = _laplace.init_targets(x.T)
        fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
        tree = _laplace.setup(sources, targets, fmm)
        values = _laplace.evaluate(tree, fmm)
        os.remove('.rhs.tmp')
        result[:] = values[:,0] / ep_in

    @bempp.api.real_callable
    def rhs2_fun(x, n, domain_index, result):
        result[0] = 0

    rhs1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
    rhs2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    return A_sys, rhs1, rhs2

def juffer_in(dirichl_space, neumann_space, assembler, q, x_q):
    ep_in = helpers.params['ep_in']
    ep_ex = helpers.params['ep_ex']
    kappa = helpers.params['kappa']

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

    A_sys = bempp.api.BlockedOperator(2, 2)
    A_sys[0, 0] = (0.5*(1.0 + ep)*phi_id) - L1
    A_sys[0, 1] = (-1.0)*L2
    A_sys[1, 0] = L3
    A_sys[1, 1] = (0.5*(1.0 + (1.0/ep))*dph_id) - L4

    @bempp.api.callable(vectorized=True)
    def rhs1_fun(x, n, domain_index, result):
        import exafmm.laplace as _laplace
        sources = _laplace.init_sources(x_q, q)
        targets = _laplace.init_targets(x.T)
        fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
        tree = _laplace.setup(sources, targets, fmm)
        values = _laplace.evaluate(tree, fmm)
        os.remove('.rhs.tmp')
        result[:] = values[:,0] / ep_in

    @bempp.api.callable(vectorized=True)
    def rhs2_fun(x, n, domain_index, result):
        import exafmm.laplace as _laplace
        sources = _laplace.init_sources(x_q, q)
        targets = _laplace.init_targets(x.T)
        fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
        tree = _laplace.setup(sources, targets, fmm)
        values = _laplace.evaluate(tree, fmm)
        os.remove('.rhs.tmp')
        result[:] = np.sum(values[:,1:] * n.T, axis=1) / ep_in

    rhs1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
    rhs2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    return A_sys, rhs1, rhs2

def juffer_ex(dirichl_space, neumann_space, assembler, q, x_q):
    ep_in = helpers.params['ep_in']
    ep_ex = helpers.params['ep_ex']
    kappa = helpers.params['kappa']

    phi_id = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
    dph_id = sparse.identity(neumann_space, neumann_space, neumann_space)
    ep = ep_ex/ep_in

    dF = laplace.double_layer(dirichl_space, dirichl_space, dirichl_space, assembler=assembler)
    dP = modified_helmholtz.double_layer(dirichl_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
    B = 1/ep * dF - dP

    F = laplace.single_layer(neumann_space, dirichl_space, dirichl_space, assembler=assembler)
    P = modified_helmholtz.single_layer(neumann_space, dirichl_space, dirichl_space, kappa, assembler=assembler)
    A = F - P

    ddF = laplace.hypersingular(dirichl_space, neumann_space, neumann_space, assembler=assembler)
    ddP = modified_helmholtz.hypersingular(dirichl_space, neumann_space, neumann_space, kappa, assembler=assembler)
    D = 1/ep * (ddP - ddF)

    dF0 = laplace.adjoint_double_layer(neumann_space, neumann_space, neumann_space, assembler=assembler)
    dP0 = modified_helmholtz.adjoint_double_layer(neumann_space, neumann_space, neumann_space, kappa, assembler=assembler)
    C = dF0 - 1.0/ep*dP0

    A_sys = bempp.api.BlockedOperator(2, 2)
    A_sys[0, 0] = (0.5*(1.0 + ep)*phi_id) + B
    A_sys[0, 1] = -A
    A_sys[1, 0] = D
    A_sys[1, 1] = (0.5*(1.0 + (1.0/ep))*dph_id) - C

    @bempp.api.callable(vectorized=True)
    def rhs1_fun(x, n, domain_index, result):
        import exafmm.laplace as _laplace
        sources = _laplace.init_sources(x_q, q)
        targets = _laplace.init_targets(x.T)
        fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
        tree = _laplace.setup(sources, targets, fmm)
        values = _laplace.evaluate(tree, fmm)
        os.remove('.rhs.tmp')
        result[:] = values[:,0] / ep_ex

    @bempp.api.callable(vectorized=True)
    def rhs2_fun(x, n, domain_index, result):
        import exafmm.laplace as _laplace
        sources = _laplace.init_sources(x_q, q)
        targets = _laplace.init_targets(x.T)
        fmm = _laplace.LaplaceFmm(p=10, ncrit=500, filename='.rhs.tmp')
        tree = _laplace.setup(sources, targets, fmm)
        values = _laplace.evaluate(tree, fmm)
        os.remove('.rhs.tmp')
        result[:] = np.sum(values[:,1:] * n.T, axis=1) / ep_ex

    rhs1 = bempp.api.GridFunction(dirichl_space, fun=rhs1_fun)
    rhs2 = bempp.api.GridFunction(neumann_space, fun=rhs2_fun)

    return A_sys, rhs1, rhs2
