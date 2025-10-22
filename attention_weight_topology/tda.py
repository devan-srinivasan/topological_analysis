import torch
import numpy as np
import networkx as nx
import gudhi
from typing import Union
from itertools import combinations

class Graph:
    def __init__(self, adj_matrix: torch.Tensor = None, names: list[str] = None):
        self.names = {}
        if adj_matrix is not None:
            self.parse_adjacency_matrix(adj_matrix, names)
        else:
            self.adj_matrix = torch.zeros((0, 0))
        self.simplicial_complex = None
        self.features:dict[int,list|set] = {}

    def add_vertex(self, name: str = None, edges: torch.Tensor = None):
        n = self.adj_matrix.shape[0]
        new_matrix = torch.zeros((n + 1, n + 1))
        new_matrix[:n, :n] = self.adj_matrix
        if edges is not None:
            if edges.shape != (n,):
                raise ValueError(f"Edges tensor must have shape ({n},) as there are currently {n} vertices.")
            new_matrix[n, :n] = edges
            # we don't add symmetrically since this is kind of a directed graph
            #   new_matrix[:n, n] = edges
        self.adj_matrix = new_matrix
        self.names[n] = name

    def add_edge(self, from_vertex, to_vertex, weight):
        self.adj_matrix[from_vertex, to_vertex] = weight
    
    def parse_adjacency_matrix(self, adj_matrix, names: list[str] = None):
        # check this is a valid graph
        # 1. make sure enough names
        if names and len(names) != adj_matrix.shape[0]:
            raise ValueError("Number of names does not match number of vertices in adjacency matrix.")
        
        self.adj_matrix = adj_matrix
        self.names = {i: names[i] if names and i < len(names) else None for i in range(adj_matrix.shape[0])}
    
    def build_simplicial_complex(self) -> list[dict]:
        """
        A manual (TODO vectorized) implementation of extracting topological features from the graph
        """
        pass

    def build_connected_components(self):
        """
        Build connected components using networkx
        """
        nx_graph = nx.from_numpy_array(self.adj_matrix.numpy())
        components = list(nx.connected_components(nx_graph))
        self.features[0] = [frozenset(comp) for comp in components]

# mat = torch.ones((4, 4)) - torch.eye(4)
# mat[1] *= 0
# mat[:, 1] *= 0

# G = Graph(
#     adj_matrix = mat,
# )