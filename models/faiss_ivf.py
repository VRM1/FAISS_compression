"""
FAISS IVF implementation for approximate vector search.
"""

import faiss
import time
import numpy as np
import os


def train_ivf_index(database_vectors, n_regions=10):
    """
    Train an IVF index (without adding vectors).
    
    Args:
        database_vectors (numpy.ndarray): Database vectors for training
        n_regions (int): Number of cluster regions
        
    Returns:
        tuple: (index, train_time) - trained index and training time
    """
    dimension = database_vectors.shape[1]
    
    # Create the index
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, n_regions, faiss.METRIC_L2)
    
    # Train the index
    train_start = time.time()
    index.train(database_vectors)
    train_time = time.time() - train_start
    
    return index, train_time


def add_vectors_to_index(index, database_vectors):
    """
    Add vectors to a trained index.
    
    Args:
        index (faiss.Index): Trained FAISS index
        database_vectors (numpy.ndarray): Vectors to add
        
    Returns:
        float: Time taken to add vectors
    """
    # Add vectors
    add_start = time.time()
    index.add(database_vectors)
    add_time = time.time() - add_start
    
    return add_time


def faiss_ivf_search(database_vectors, query_vectors, k=5, n_regions=10, nprobe=3, 
                    trained_index=None):
    """
    Perform FAISS IVF search.
    
    Args:
        database_vectors (numpy.ndarray): Database vectors to search in
        query_vectors (numpy.ndarray): Query vectors to search for
        k (int): Number of nearest neighbors to retrieve
        n_regions (int): Number of cluster regions
        nprobe (int): Number of regions to search
        trained_index (faiss.Index, optional): Pre-trained FAISS index
        
    Returns:
        tuple: (indices, search_time, train_time, add_time) - top k matching indices, search time,
            training time, and time to add vectors
    """
    dimension = database_vectors.shape[1]
    
    # Use provided index or create a new one
    if trained_index is not None:
        index = trained_index
        train_time = 0  # Index already trained
    else:
        # Create and train index
        index, train_time = train_ivf_index(database_vectors, n_regions)
    
    # Add vectors if needed
    if index.ntotal == 0:
        add_time = add_vectors_to_index(index, database_vectors)
    else:
        add_time = 0  # Vectors already added
    
    # Set number of regions to search
    index.nprobe = nprobe
    
    # Reshape query if it's a single vector
    if len(query_vectors.shape) == 1:
        query_vectors = query_vectors.reshape(1, -1)
    
    # Perform search
    search_start = time.time()
    distances, indices = index.search(query_vectors, k)
    search_time = time.time() - search_start
    
    # If only one query, return first result
    if len(query_vectors) == 1:
        return indices[0], search_time, train_time, add_time
    else:
        return indices, search_time, train_time, add_time


def get_searched_regions(query_vector, centroids, nprobe):
    """
    Get the regions that would be searched for a query.
    
    Args:
        query_vector (numpy.ndarray): Query vector
        centroids (numpy.ndarray): Centroid vectors
        nprobe (int): Number of regions to search
        
    Returns:
        numpy.ndarray: Indices of regions that would be searched
    """
    # Ensure query is properly shaped
    if len(query_vector.shape) == 1:
        query_vector = query_vector.reshape(1, -1)
        
    # Create a temporary quantizer
    quantizer = faiss.IndexFlatL2(query_vector.shape[1])
    quantizer.add(centroids)
    
    # Find the closest nprobe regions
    _, searched_regions = quantizer.search(query_vector, nprobe)
    
    return searched_regions[0]  # Return the first batch result


def save_ivf_index(index, database_vectors, filepath, metadata=None):
    """
    Save a FAISS IVF index to disk.
    
    Args:
        index (faiss.Index): FAISS index to save
        database_vectors (numpy.ndarray): Database vectors used in the index
        filepath (str): Path to save the index
        metadata (dict, optional): Additional metadata to save with the index
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save index
    faiss.write_index(index, filepath)
    
    # Save vectors alongside index (for reference)
    np.save(f"{filepath}_vectors.npy", database_vectors)
    
    # Save metadata if provided
    if metadata:
        np.save(f"{filepath}_metadata.npy", metadata)
    
    print(f"Saved FAISS IVF index to {filepath}")


def load_ivf_index(filepath, set_nprobe=None):
    """
    Load a FAISS IVF index from disk.
    
    Args:
        filepath (str): Path to the saved index
        set_nprobe (int, optional): Set nprobe value after loading
        
    Returns:
        tuple: (index, database_vectors, metadata) - loaded index, vectors, and metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Index file not found: {filepath}")
        
    # Load index
    index = faiss.read_index(filepath)
    
    # Set nprobe if provided
    if set_nprobe is not None:
        index.nprobe = set_nprobe
    
    # Load vectors if available
    vectors_path = f"{filepath}_vectors.npy"
    if os.path.exists(vectors_path):
        database_vectors = np.load(vectors_path)
    else:
        database_vectors = None
        print(f"Warning: Vector data file not found at {vectors_path}")
    
    # Load metadata if available
    metadata_path = f"{filepath}_metadata.npy"
    if os.path.exists(metadata_path):
        try:
            metadata = np.load(metadata_path, allow_pickle=True).item()
        except:
            metadata = None
            print(f"Warning: Could not load metadata from {metadata_path}")
    else:
        metadata = None
    
    return index, database_vectors, metadata