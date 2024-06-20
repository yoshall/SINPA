# coding=utf-8
import numpy as np
import torch
import scipy.sparse as sp

import src.utils.graph_algo as graph_algo


class RandomSampler:
    """
    Sampling the input graph data.

    Args:
        adj_mat (array-like): The adjacency matrix of the graph.
        filter_type (str): The type of filter to be applied during sampling.

    Attributes:
        _adj_mat (array-like): The adjacency matrix of the graph.
        _filter_type (str): The type of filter to be applied during sampling.
    """

    def __init__(self, adj_mat, filter_type):
        self._adj_mat = adj_mat
        self._filter_type = filter_type

    def sample(self, percent):
        """
        Randomly drop edge and preserve percent% edges.

        Args:
            percent (float): The percentage of edges to preserve.

        Returns:
            array-like: The sampled adjacency matrix.
        """
        if percent >= 1.0:
            raise ValueError

        adj_sp = sp.coo_matrix(self._adj_mat)

        nnz = adj_sp.nnz
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * percent)
        perm = perm[:preserve_nnz]
        r_adj = sp.coo_matrix(
            (adj_sp.data[perm], (adj_sp.row[perm], adj_sp.col[perm])),
            shape=adj_sp.shape,
        )
        return r_adj.todense()


class CutEdgeSampler:
    """Sampling the input graph data.

    This class implements a sampling technique for graph data. It calculates the drop rate
    for each edge in the input adjacency matrix based on the out-degree of the nodes. The
    drop rate is used to create a drop mask, which is then applied to the adjacency matrix
    to obtain the sampled graph.

    Args:
        adj_mat (numpy.ndarray): The input adjacency matrix.
        filter_type (str): The type of filter to be applied.
        m (int, optional): The drop rate multiplier. Defaults to 200.

    Attributes:
        droprate (numpy.ndarray): The drop rate matrix.

    """

    def __init__(self, adj_mat, filter_type, m=200):
        self._adj_mat = adj_mat.copy()
        self._filter_type = filter_type

        new_adj = adj_mat + np.eye(adj_mat.shape[0])
        rw_adj = graph_algo.calculate_random_walk_matrix(new_adj).todense()

        square_adj = np.power(rw_adj, 2)
        out_degree = np.array(square_adj.sum(1)).flatten()

        adj_sp = sp.coo_matrix(rw_adj)

        out_degree_sum = 0
        for i in range(adj_sp.nnz):
            out_degree_sum += out_degree[adj_sp.row[i]] + out_degree[adj_sp.col[i]]

        p = np.zeros(adj_sp.nnz)
        for i in range(adj_sp.nnz):
            p[i] = (
                (out_degree[adj_sp.row[i]] + out_degree[adj_sp.col[i]])
                / out_degree_sum
                * m
            )
        self.droprate = sp.coo_matrix(
            (p, (adj_sp.row, adj_sp.col)), shape=adj_sp.shape
        ).todense()

    def sample(self, m=200):
        """Sample the graph data.

        This method applies the drop mask to the adjacency matrix to obtain the sampled graph.

        Args:
            m (int, optional): The drop rate multiplier. Defaults to 200.

        Returns:
            numpy.ndarray: The sampled adjacency matrix.

        """
        num_nodes = self._adj_mat.shape[0]
        prob = np.random.rand(num_nodes, num_nodes)
        drop_mask = (prob > self.droprate).astype(int)  # preserve rate
        return self._adj_mat * drop_mask
