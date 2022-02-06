import math
import numpy as np
import scipy.sparse as sp
from scipy import integrate
import torch

def calc_gso(dir_adj, gso_type):
    if sp.issparse(dir_adj):
        id = sp.identity(dir_adj.shape[0], format='csc')
        # Symmetrizing an adjacency matrix
        adj = dir_adj + dir_adj.T.multiply(dir_adj.T > dir_adj) - dir_adj.multiply(dir_adj.T > dir_adj)
        #adj = 0.5 * (dir_adj + dir_adj.transpose())
    
        if gso_type == 'sym_renorm_adj' or 'rw_renorm_adj' or 'sym_renorm_lap' or 'rw_renorm_lap':
            adj = adj + id
    
        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = sp.diags(row_sum_inv_sqrt, format='csc')
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj
        
        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = adj.sum(axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = sp.diags(row_sum_inv, format='csc')
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')
    
    else:
        id = np.identity(dir_adj.shape[0])
        # Symmetrizing an adjacency matrix
        adj = np.maximum(dir_adj, dir_adj.T)
        #adj = 0.5 * (dir_adj + dir_adj.T)

        if gso_type == 'sym_renorm_adj' or 'rw_renorm_adj' or 'sym_renorm_lap' or 'rw_renorm_lap':
            adj = adj + id

        if gso_type == 'sym_norm_adj' or gso_type == 'sym_renorm_adj' \
            or gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
            row_sum = np.sum(adj, axis=1)
            row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.
            deg_inv_sqrt = np.diag(row_sum_inv_sqrt)
            # A_{sym} = D^{-0.5} * A * D^{-0.5}
            sym_norm_adj = deg_inv_sqrt.dot(adj).dot(deg_inv_sqrt)

            if gso_type == 'sym_norm_lap' or gso_type == 'sym_renorm_lap':
                sym_norm_lap = id - sym_norm_adj
                gso = sym_norm_lap
            else:
                gso = sym_norm_adj

        elif gso_type == 'rw_norm_adj' or gso_type == 'rw_renorm_adj' \
            or gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
            row_sum = np.sum(adj, axis=1).A1
            row_sum_inv = np.power(row_sum, -1)
            row_sum_inv[np.isinf(row_sum_inv)] = 0.
            deg_inv = np.diag(row_sum_inv)
            # A_{rw} = D^{-1} * A
            rw_norm_adj = deg_inv.dot(adj)

            if gso_type == 'rw_norm_lap' or gso_type == 'rw_renorm_lap':
                rw_norm_lap = id - rw_norm_adj
                gso = rw_norm_lap
            else:
                gso = rw_norm_adj

        else:
            raise ValueError(f'{gso_type} is not defined.')

    return gso

def calc_hkgcn_feature(gso, feature, t, K):
    gso = 0.5 * gso
    t_ = 2 * t

    # The Chebyshev polynomials of the first kind
    T = []
    # The modified Bessel function of the first kind
    B = []

    T_0 = feature
    T.append(T_0)
    mod_bessel_0 = lambda theta: math.exp(t_ * math.cos(theta))
    B_0 = 1 / math.pi * integrate.quad(mod_bessel_0, 0, math.pi)[0]
    B.append(B_0)
    X_t = B_0 * T_0

    T_1 = gso.dot(feature)
    T.append(T_1)
    mod_bessel = lambda theta, n: math.exp(t_ * math.cos(theta)) * math.cos(n * theta)
    B_1 = 1 / math.pi * integrate.quad(mod_bessel, 0, math.pi, args=(1,))[0]
    B.append(B_1)
    X_t = X_t - 2 * B_1 * T_1

    for k in range(2, K):
        T.append(2 * gso.dot(T[k - 1]) - T[k - 2])
        B.append(1 / math.pi * integrate.quad(mod_bessel, 0, math.pi, args = (k,))[0])
        if k % 2 == 0:
            X_t = X_t + 2 * B[k] * T[k]
        else:
            X_t = X_t - 2 * B[k] * T[k]

    return X_t

def cnv_sparse_mat_to_coo_tensor(sp_mat, device):
    # convert a compressed sparse row (csr) or compressed sparse column (csc) matrix to a hybrid sparse coo tensor
    sp_coo_mat = sp_mat.tocoo()
    i = torch.from_numpy(np.vstack((sp_coo_mat.row, sp_coo_mat.col)))
    v = torch.from_numpy(sp_coo_mat.data)
    s = torch.Size(sp_coo_mat.shape)

    if sp_mat.dtype == np.complex64 or sp_mat.dtype == np.complex128:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.complex64, device=device, requires_grad=False)
    elif sp_mat.dtype == np.float32 or sp_mat.dtype == np.float64:
        return torch.sparse_coo_tensor(indices=i, values=v, size=s, dtype=torch.float32, device=device, requires_grad=False)
    else:
        raise TypeError(f'ERROR: The dtype of {sp_mat} is {sp_mat.dtype}, not been applied in implemented models.')

def calc_accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double().sum()
    accuracy = correct / len(labels)

    return accuracy