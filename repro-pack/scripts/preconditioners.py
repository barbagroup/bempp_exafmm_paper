import helpers
import bempp.api
from bempp.api.operators.boundary import sparse, laplace, modified_helmholtz
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import aslinearoperator

def block_diagonal(dirichl_space, neumann_space, A):
    ep_in = helpers.PARAMS.ep_in
    ep_ex = helpers.PARAMS.ep_ex
    kappa = helpers.PARAMS.kappa

    identity = sparse.identity(dirichl_space, dirichl_space, dirichl_space)
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

    block_mat_precond = bmat([[diags(diag11_inv), diags(diag12_inv)],
                                [diags(diag21_inv), diags(diag22_inv)]]).tocsr()
    return aslinearoperator(block_mat_precond)
