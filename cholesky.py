"""High-dimensional sparse embeddings for collaborative filtering,
using Cholesky factorization and pruning.
"""
import numpy as np
import scipy as sp


def cholesky_embeddings(x, l2_reg=500.0, beta=2.0, row_nnz=None, sort_by_nn=False):
    """Cholesky factors of -P + beta * I

    Args:
    - x (sp.sparse.csr_matrix), sparse user-item matrix
    - l2_reg (float): l2 regularization coefficient
    - beta (float): cholesky embeddings hyper-parameter, see paper
    - row_nnz (int): the desired number of non-zeros per embedding (None for dense)
    - sort_by_nn (bool): sort by number of "neighbors" before factorizing (see paper)
    """
    assert beta > 1.0

    print('computing gramm matrix G')
    G = x.T @ x
    diag_indices = np.diag_indices(G.shape[0])
    G[diag_indices] += l2_reg
    if sort_by_nn:
        print('sorting items by number of neighbors')
        items_by_nn = np.argsort(np.ravel(G.getnnz(axis=0)))  # nn = non-zero co-counts
        G = G[items_by_nn][:, items_by_nn]
    print('computing inverse P')
    P = np.linalg.inv(G.toarray())
    diag_P = np.diag(P)

    print('factorizing -P + beta * diag_P')
    A = -P
    A[diag_indices] += beta * diag_P
    E = sp.linalg.cholesky(A, lower=True)
    print('pruning factors')
    E = prune_rows(x, target_nnz=row_nnz)

    print('computing priors')
    prior = 1 / diag_P
    prior[diag_P == 0] = 0.0
    if sort_by_nn:
        print('undo sort items')
        original_order = np.argsort(items_by_nn)
        prior = prior[original_order]
        E = E[original_order]

    return E, prior


def prune_rows(x, target_nnz=100):
    """Prune the rows of 2d np.array by setting elements with low absolute
    value to 0.0 and return a sp.sparse.csr_matrix.

    Args:
    - x (np.array), array to be pruned
    - target_nnz (float): the desired number of non-zeros per row
    """
    target_density = target_nnz / x.shape[1]
    pruned_rows = [prune(row, target_density=target_density) for row in x]

    return sp.sparse.vstack(pruned_rows)


def prune(x, target_density=0.01):
    """Prune a np.array by setting elements with low absolute value
    to 0.0 and return as a sp.sparse.csr_matrix.

    Args:
    - x (np.array), array to be pruned
    - target_density (float): the desired overall fraction of non-zeros
    """
    thr = np.quantile(np.abs(x), 1.0 - target_density)
    x[np.abs(x) < thr] = 0.0

    return sp.sparse.csr_matrix(x)
