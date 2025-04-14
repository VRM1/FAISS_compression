"""
Visualization utilities for displaying vector spaces and search results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import faiss


def plot_vectors_3d(vectors, query_vector=None, matches=None, title="Vector Space Visualization", 
                   save_path=None):
    """
    Basic 3D visualization of vectors without regions.
    
    Args:
        vectors (numpy.ndarray): Database vectors to plot
        query_vector (numpy.ndarray, optional): Query vector to highlight
        matches (numpy.ndarray, optional): Indices of matching vectors to highlight
        title (str): Plot title
        save_path (str, optional): Path to save the figure, if specified
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all vectors
    ax.scatter(vectors[:, 0], vectors[:, 1], vectors[:, 2], c='blue', alpha=0.5, label='Database vectors')
    
    # Plot query vector if provided
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=100, label='Query vector')
    
    # Plot matches if provided
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compute_vector_assignments(vectors, centroids):
    """
    Compute which vectors belong to which centroids.
    
    Args:
        vectors (numpy.ndarray): Vectors to assign
        centroids (numpy.ndarray): Centroid vectors
        
    Returns:
        tuple: (distances, assignments) - arrays of distance to centroid and cluster assignment
    """
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(centroids)
    distances, assignments = index.search(vectors, 1)
    return distances, assignments.ravel()


def train_kmeans_get_centroids(vectors, n_clusters):
    """
    Train k-means and get centroids.
    
    Args:
        vectors (numpy.ndarray): Vectors to cluster
        n_clusters (int): Number of clusters
        
    Returns:
        numpy.ndarray: Centroid vectors
    """
    kmeans = faiss.Kmeans(d=vectors.shape[1], k=n_clusters, niter=20, verbose=False)
    kmeans.train(vectors)
    return kmeans.centroids


def plot_vectors_with_regions(vectors, centroids, query_vector=None, matches=None, 
                            searched_regions=None, title="Vector Space with FAISS Regions",
                            save_path=None):
    """
    Visualize vectors in 3D space with their clusters/regions.
    
    Args:
        vectors (numpy.ndarray): Database vectors to plot
        centroids (numpy.ndarray): Centroid vectors defining regions
        query_vector (numpy.ndarray, optional): Query vector to highlight
        matches (numpy.ndarray, optional): Indices of matching vectors to highlight
        searched_regions (numpy.ndarray, optional): Indices of regions that were searched
        title (str): Plot title
        save_path (str, optional): Path to save the figure, if specified
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Assign each vector to nearest centroid
    distances, assignments = compute_vector_assignments(vectors, centroids)
    
    # Plot vectors colored by their cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, len(centroids)))
    for i in range(len(centroids)):
        cluster_vectors = vectors[assignments == i]
        if len(cluster_vectors) > 0:
            # Make vectors transparent if their region wasn't searched
            alpha = 1.0 if searched_regions is None or i in searched_regions else 0.1
            ax.scatter(cluster_vectors[:, 0], cluster_vectors[:, 1], cluster_vectors[:, 2], 
                      c=[colors[i]], alpha=alpha, label=f'Region {i}')
    
    # Plot centroids
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
              c='black', s=100, marker='*', label='Region Centers')
    
    # Plot query vector
    if query_vector is not None:
        ax.scatter(query_vector[0], query_vector[1], query_vector[2], 
                  c='red', s=200, marker='x', label='Query Vector')
    
    # Plot matches
    if matches is not None:
        match_vectors = vectors[matches]
        ax.scatter(match_vectors[:, 0], match_vectors[:, 1], match_vectors[:, 2], 
                  c='green', s=100, label='Matches')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_performance_comparison(results, save_path=None):
    """
    Plot performance comparison between different search methods.
    
    Args:
        results (dict): Dictionary containing performance results
        save_path (str, optional): Path to save the figure, if specified
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results['sizes'], results['brute_force'], 'o-', label='Brute Force Cosine')
    plt.plot(results['sizes'], results['faiss_flat'], 's-', label='FAISS Flat L2')
    plt.plot(results['sizes'], results['faiss_ivf'], '^-', label='FAISS IVF')

    plt.xscale('log')  # Log scale for vector sizes
    plt.yscale('log')  # Log scale for times

    plt.xlabel('Number of Vectors')
    plt.ylabel('Average Search Time per Query (seconds)')
    plt.title('Search Time Comparison: Brute Force vs FAISS Methods')
    plt.grid(True)
    plt.legend()

    # Add value annotations
    for i, size in enumerate(results['sizes']):
        plt.annotate(f'{results["brute_force"][i]:.6f}', 
                    (size, results['brute_force'][i]), 
                    textcoords="offset points", xytext=(0,10))
        plt.annotate(f'{results["faiss_flat"][i]:.6f}', 
                    (size, results['faiss_flat'][i]), 
                    textcoords="offset points", xytext=(0,-15))
        plt.annotate(f'{results["faiss_ivf"][i]:.6f}', 
                    (size, results['faiss_ivf'][i]), 
                    textcoords="offset points", xytext=(0,10))

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()