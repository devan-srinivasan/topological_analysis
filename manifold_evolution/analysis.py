import gudhi
import torch
import numpy as np

def rips(
        data, 
        max_edge_length: float, 
        max_dimension: int = 3
    ) -> gudhi.SimplexTree:
    """
    Easy wrapper to get a simplex tree with persistence on a point cloud.
        Call .betti_numbers() on the returned simplex tree to get Betti numbers.
        Call .persistence_pairs() to get persistence pairs.
    """
    rips_complex = gudhi.RipsComplex(points=data, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
    simplex_tree.compute_persistence()
    return simplex_tree

def wasserstein_distance(
        persistence1,
        persistence2
    ) -> float:
    """
    Compute the Wasserstein distance between two persistence diagrams.
    - persistence1, persistence2: persistence diagrams as (:, 2) numpy arrays or tensors
    """
    return gudhi.wasserstein.wasserstein_distance(persistence1, persistence2, order=1, internal_p=2, enable_autodiff=False)

def build_graph(
        points: torch.Tensor,
        max_edge_length: float = float('inf'),
    ) -> torch.Tensor:
    """
    Build a weighted graph adjacency matrix from a point cloud. Helper for other functions
    - points: (N, D) tensor of N points in D dimensions
    - max_edge_length: maximum distance between points to create an edge. Default is inf (fully connected graph)
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points, points, p=2)
    # Create adjacency matrix
    adj_matrix = (dist_matrix < max_edge_length).float()
    return adj_matrix

def build_local_linear_approximations(
        points: torch.Tensor,
        k: int = 5,
    ) -> torch.Tensor:
    """
    Builds local linear approximations of the dataset's manifold at each KNN cluster.
    - points: (N, D) tensor of N points in D dimensions
    - k: number of nearest neighbors to consider for local linear approximation (k+1 points in the approximation)
    Returns:
        local_bases: (N, D, D) tensor of local basis vectors at each point
        estimated_dimensions: (N,) tensor of estimated local dimensions at each point

    We use PCA to estimate the local tangent space at each point neighbourhood.
    """

    # Step 1: Do KNN to get indices tensor shape (N, k) representing the neighbourhood of each point
    dist_matrix = torch.cdist(points, points, p=2)
    dist_matrix.fill_diagonal_(float('inf'))  # Exclude self-distances
    _, knn_indices = torch.topk(dist_matrix, k, largest=False)

    # Step 2: For each row in knn_indices perform PCA on points[row] to get local basis. 
    #         Estimate the rank of the local covariance matrix to get local dimension, but
    #         keep all basis vectors for now.
    N, D = points.shape
    local_bases = torch.zeros((N, D, D))
    estimated_dimensions = []
    for i in range(N):
        neighborhood = points[knn_indices[i]]  # (k, D)
        centered_neighborhood = neighborhood - neighborhood.mean(dim=0)
        U, S, Vt = torch.linalg.svd(centered_neighborhood, full_matrices=False)
        local_bases[i] = Vt.T  # Store all basis vectors

        # Estimate local dimension using a threshold on singular values
        threshold = 1e-3
        estimated_dim = (S > threshold).sum().item()
        estimated_dimensions.append(estimated_dim)

    estimated_dimensions = torch.tensor(estimated_dimensions)
    return local_bases, estimated_dimensions

