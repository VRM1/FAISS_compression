"""
Brute force cosine similarity search implementation.
"""

import numpy as np
import time


def brute_force_cosine_search(database_vectors, query_vector, k=5):
    """
    Perform brute force cosine similarity search.
    
    Args:
        database_vectors (numpy.ndarray): Database vectors to search in
        query_vector (numpy.ndarray): Query vector to search for
        k (int): Number of nearest neighbors to retrieve
        
    Returns:
        tuple: (indices, search_time) - top k matching indices and search time
    """
    start_time = time.time()
    
    # Ensure query vector is 1D
    if len(query_vector.shape) > 1:
        query_vector = query_vector.ravel()
    
    # Calculate cosine similarity
    similarities = np.dot(database_vectors, query_vector)
    
    # Get top k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    end_time = time.time()
    search_time = end_time - start_time
    
    return top_k_indices, search_time