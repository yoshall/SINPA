import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import torch

from scipy.sparse import linalg


def calculate_normalized_laplacian(adj):
    """
    Calculate the normalized Laplacian matrix of an adjacency matrix.

    The normalized Laplacian matrix is defined as L = I - D^-1/2 A D^-1/2,
    where A is the adjacency matrix and D is the degree matrix.

    :param adj: The adjacency matrix.
    :return: The normalized Laplacian matrix.
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d + 1e-6, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = (
        sp.eye(adj.shape[0])
        - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    )
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    """
    Calculates the random walk matrix for a given adjacency matrix.

    Parameters:
    - adj_mx (numpy.ndarray or scipy.sparse.coo_matrix): The adjacency matrix.

    Returns:
    - scipy.sparse.coo_matrix: The random walk matrix.
    """
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d + 1e-6, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    """
    Calculates the reverse random walk matrix of the given adjacency matrix.

    Parameters:
    adj_mx (numpy.ndarray): The adjacency matrix.

    Returns:
    numpy.ndarray: The reverse random walk matrix.
    """
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    """
    Calculates the scaled Laplacian matrix for a given adjacency matrix.

    Parameters:
        adj_mx (numpy.ndarray): The adjacency matrix.
        lambda_max (float): The maximum eigenvalue of the Laplacian matrix. If None, it will be calculated.
        undirected (bool): Whether the adjacency matrix is undirected. If True, the matrix will be symmetrized.

    Returns:
        numpy.ndarray: The scaled Laplacian matrix.

    """
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which="LM")
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format="csr", dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def load_graph_data(pkl_filename):
    """
    Load graph data from a pickle file.

    Args:
        pkl_filename (str): The path to the pickle file.

    Returns:
        tuple: A tuple containing the sensor IDs, sensor ID to index mapping, and the adjacency matrix.
    """
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    adj_mx = adj_mx - np.eye(adj_mx.shape[0])
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    """
    Load data from a pickle file.

    Parameters:
    pickle_file (str): The path to the pickle file.

    Returns:
    The loaded data from the pickle file.
    """
    try:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, "rb") as f:
            pickle_data = pickle.load(f, encoding="latin1")
    except Exception as e:
        print("Unable to load data ", pickle_file, ":", e)
        raise
    return pickle_data


def calculate_cheb_poly(L, Ks):
    """
    Calculate Chebyshev polynomials up to Ks-1 order.

    Parameters:
    - L: numpy.ndarray
        The Laplacian matrix of the graph.
    - Ks: int
        The number of Chebyshev polynomials to calculate.

    Returns:
    - numpy.ndarray
        An array of Chebyshev polynomials up to Ks-1 order.
    """
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def sym_adj(adj):
    """
    Symmetrically normalize adjacency matrix.

    Parameters:
    - adj: numpy.ndarray or scipy.sparse matrix
        The adjacency matrix to be normalized.

    Returns:
    - numpy.ndarray
        The symmetrically normalized adjacency matrix.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum + 1e-6, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return (
        adj.dot(d_mat_inv_sqrt)
        .transpose()
        .dot(d_mat_inv_sqrt)
        .astype(np.float32)
        .todense()
    )


def asym_adj(adj):
    """
    Compute the asymmetric adjacency matrix.

    Parameters:
    adj (numpy.ndarray or scipy.sparse.coo_matrix): The input adjacency matrix.

    Returns:
    numpy.ndarray: The computed asymmetric adjacency matrix.

    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum + 1e-6, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.0
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


# def _generate_G_from_H(H, variable_weight=False):
#     """
#     Calculate G from hypergraph incidence matrix H.

#     Args:
#         H (torch.Tensor): Hypergraph incidence matrix H.
#         variable_weight (bool, optional): Whether the weight of hyperedge is variable. Defaults to False.

#     Returns:
#         torch.Tensor: G, the calculated graph.

#     """
#     n_edge = H.shape[1]
#     # the weight of the hyperedge
#     W = torch.ones(n_edge).cuda()  # [n_edge]
#     # the degree of the node
#     DV = torch.sum(H * W, axis=1)  # [n_nodes]
#     # the degree of the hyperedge
#     DE = torch.sum(H, axis=0)  # [n_edge]

#     invDE = torch.diag(torch.pow(DE + 1e-6, -1))  # [n_edge, n_edge]
#     DV2 = torch.diag(torch.pow(DV + 1e-6, -0.5))  # [n_node, n_node]
#     W = torch.diag(W)  # [n_edge, n_edge]
#     H = H  # [n_node, n_edge]
#     HT = H.T  # [n_edge, n_node]

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         return G


def _generate_G_from_H(H, variable_weight=False):
    """
    Calculate G from hypergraph incidence matrix H.

    :param H: Hypergraph incidence matrix H.
    :param variable_weight: Whether the weight of hyperedge is variable.
    :return: G, the calculated graph.
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE + 1e-6, -1)))
    DV2 = np.mat(np.diag(np.power(DV + 1e-6, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return np.asarray(G)
