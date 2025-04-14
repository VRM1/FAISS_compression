"""
FAISS flat index implementation for basic L2 search.
"""

import faiss
import time
import numpy as np
import os


def faiss_flat_l2_search(database_vectors, query_vector, k=5):
    """
    Perform basic FAISS L2 search (no regions).
    
    Args:
        database_vectors (numpy.ndarray): Database vectors to search in
        query_vector (numpy.ndarray): Query vector to search for
        k (int): Number of nearest neighbors to retrieve
        
    Returns:
        tuple: (indices, search_time) - top k matching indices and search time
    """
    dimension = database_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    start_time = time.time()
    
    # Add vectors to index
    index.add(database_vectors)
    
    # Ensure query vector is properly shaped (batch dimension)
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Search
    distances, indices = index.search(query_vector, k)
    
    end_time = time.time()
    search_time = end_time - start_time
    
    return indices[0], search_time


def save_flat_index(index, database_vectors, filepath):
    """
    Save a FAISS flat index to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        database_vectors (numpy.ndarray): Database vectors used in the index
        filepath (str): Path to save the index
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save index
    faiss.write_index(index, filepath)
    
    # Save vectors alongside index (for reference)
    np.save(f"{filepath}_vectors.npy", database_vectors)
    
    print(f"Saved FAISS flat index to {filepath}")


def load_flat_index(filepath):
    """
    Load a FAISS flat index from disk.
    
    Args:
        filepath (str): Path to the saved index
        
    Returns:
        tuple: (index, database_vectors) - loaded index and vectors
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index file not found: {filepath}")
        
    # Load index
    index = faiss.read_index(filepath)
    
    # Load vectors if available
    vectors_path = f"{filepath}_vectors.npy"
    if os.path.exists(vectors_path):
        database_vectors = np.load(vectors_path)
    else:
        database_vectors = None
        print(f"Warning: Vector data file not found at {vectors_path}")
    
    return index, database_vectors