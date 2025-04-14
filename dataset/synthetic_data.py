"""
Functions for generating synthetic vector data for FAISS experiments.
"""

import numpy as np
from sklearn.preprocessing import normalize


def generate_sample_vectors(n_vectors=100, n_dimensions=3, seed=42):
    """
    Generate synthetic vectors for demonstration.
    
    Args:
        n_vectors (int): Number of vectors to generate
        n_dimensions (int): Dimensionality of vectors
        seed (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Normalized vectors with shape (n_vectors, n_dimensions)
    """
    np.random.seed(seed)
    vectors = np.random.randn(n_vectors, n_dimensions)
    normalized_vectors = normalize(vectors, norm='l2')
    return normalized_vectors


def create_query_vector(n_dimensions=3, seed=42):
    """
    Create a random query vector.
    
    Args:
        n_dimensions (int): Dimensionality of vector
        seed (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: A single normalized vector with shape (n_dimensions,)
    """
    np.random.seed(seed)
    query = np.random.randn(1, n_dimensions)
    normalized_query = normalize(query, norm='l2')[0]
    return normalized_query