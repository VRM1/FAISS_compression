"""
FAISS clustering tool for vector dimensionality reduction.
Converts a large set of vectors into a smaller set of representative centroids.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import faiss
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import from dataset module
from dataset.parquet_loader import load_data_from_config


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_faiss_index(vectors, config):
    """Create and train a FAISS index based on configuration."""
    d = vectors.shape[1]  # Dimension
    n_clusters = config['clustering']['n_clusters']
    use_pq = config['clustering']['use_pq']
    
    print(f"Creating FAISS index with {n_clusters} clusters...")
    
    # Create appropriate quantizer and index based on configuration
    if config['clustering']['gpu'] and faiss.get_num_gpus() > 0:
        print("Using GPU for clustering...")
        try:
            res = faiss.StandardGpuResources()
            quantizer = faiss.IndexFlatL2(d)
            quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)
        except Exception as e:
            print(f"GPU initialization failed: {e}. Falling back to CPU.")
            quantizer = faiss.IndexFlatL2(d)
    else:
        quantizer = faiss.IndexFlatL2(d)
    
    if use_pq:
        # Create an index with Product Quantization
        m = config['clustering']['pq_m']  # Number of sub-quantizers
        bits = config['clustering']['pq_bits']  # Bits per sub-quantizer
        
        print(f"Using Product Quantization with {m} sub-quantizers and {bits} bits each")
        
        # Ensure m divides the dimension properly
        if d % m != 0:
            print(f"Warning: PQ factor {m} does not divide dimension {d} evenly.")
            
        index = faiss.IndexIVFPQ(quantizer, d, n_clusters, m, bits)
    else:
        # Create a flat index without compression
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
    
    # Configure verbose output
    index.verbose = config['clustering']['verbose']
    
    # Determine sample size for training
    sample_size = config['training']['sample_size']
    if sample_size < 0 or sample_size > vectors.shape[0]:
        sample_size = vectors.shape[0]
    
    # Create training sample
    sample_indices = np.random.choice(vectors.shape[0], min(sample_size, vectors.shape[0]), replace=False)
    train_vectors = vectors[sample_indices]
    
    print(f"Training index on {train_vectors.shape[0]} vectors...")
    
    # Set up progress monitoring with tqdm
    original_train = index.train
    
    def train_with_pbar(x):
        pbar = tqdm(total=100, desc="Training progress")
        last_value = [0]  # Use list for mutable reference
        
        def update_pbar():
            if hasattr(index, '_train_progress') and index._train_progress > last_value[0]:
                pbar.update(index._train_progress - last_value[0])
                last_value[0] = index._train_progress
                
        # Set up a hook to update the progress bar
        if hasattr(index, 'set_progress_callback'):
            index.set_progress_callback(lambda x: update_pbar())
        
        # Start timer
        start_time = time.time()
        original_train(x)
        end_time = time.time()
        
        pbar.close()
        return end_time - start_time
    
    # If progress callback is supported, use it
    if hasattr(index, 'set_progress_callback'):
        train_time = train_with_pbar(train_vectors)
    else:
        # Fallback to standard training with just a timing display
        start_time = time.time()
        index.train(train_vectors)
        train_time = time.time() - start_time

    print(f"Training completed in {train_time:.2f} seconds")
    
    return index


def add_vectors_to_index(index, vectors):
    """Add vectors to the index with progress display."""
    print(f"Adding {vectors.shape[0]} vectors to index...")
    
    # Add with progress bar
    batch_size = 10000  # Process in batches to show progress
    num_batches = (vectors.shape[0] + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    with tqdm(total=vectors.shape[0], desc="Adding vectors") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, vectors.shape[0])
            batch = vectors[start_idx:end_idx]
            
            index.add(batch)
            pbar.update(batch.shape[0])
    
    add_time = time.time() - start_time
    print(f"Added all vectors in {add_time:.2f} seconds")
    
    return add_time


def extract_codebook(index, config, vectors=None):
    """Extract the codebook (cluster centroids) from the index."""
    n_clusters = config['clustering']['n_clusters']
    dimension = index.d
    
    print("Extracting codebook...")
    
    # For GPU indexes, convert to CPU first for easier access
    if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
        print("Converting GPU index to CPU for codebook extraction...")
        cpu_index = faiss.index_gpu_to_cpu(index)
        
        # Extract centroids from CPU index
        if hasattr(cpu_index, 'quantizer'):
            if hasattr(cpu_index.quantizer, 'centroids'):
                # Direct access to centroids
                centroids = faiss.vector_to_array(cpu_index.quantizer.centroids)
                centroids = centroids.reshape(n_clusters, dimension)
            else:
                # For more complex quantizers that don't expose centroids directly
                print("Extracting centroids from complex quantizer...")
                centroids = np.zeros((n_clusters, dimension), dtype='float32')
                
                # Try to use reconstruct method if available
                if hasattr(cpu_index.quantizer, 'reconstruct'):
                    for i in range(n_clusters):
                        try:
                            cpu_index.quantizer.reconstruct(i, centroids[i])
                        except RuntimeError as e:
                            print(f"Warning: Failed to reconstruct centroid {i}: {e}")
                            # Fall back to random initialization if reconstruct fails
                            centroids[i] = np.random.rand(dimension).astype('float32')
                else:
                    print("Warning: Could not find direct method to extract centroids. Using alternative approach.")
                    # Alternative approach: Use the quantizer to find the closest centroid for synthetic points
                    # Generate random points and see which centroids they map to
                    test_vectors = np.random.rand(n_clusters * 10, dimension).astype('float32')
                    _, assignments = cpu_index.quantizer.search(test_vectors, 1)
                    
                    # Use assignments to count occurrences of each centroid
                    unique_centroids = np.unique(assignments.flatten())
                    print(f"Found {len(unique_centroids)} unique centroids out of {n_clusters} expected")
                    
                    # If we couldn't find all centroids, we'll need to use a different approach
                    # In this case, we'll use k-means on the original vectors to approximate centroids
                    if len(unique_centroids) < n_clusters:
                        print("Warning: Could not extract all centroids. Using k-means approximation.")
                        # Use sklearn's KMeans as a fallback
                        try:
                            from sklearn.cluster import KMeans
                            # Get a sample of vectors for k-means
                            if vectors is not None:
                                kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=20).fit(vectors[:min(len(vectors), 100000)])
                                centroids = kmeans.cluster_centers_.astype('float32')
                            else:
                                print("Error: No vectors available for k-means fallback.")
                                centroids = np.random.rand(n_clusters, dimension).astype('float32')
                        except ImportError:
                            print("Error: sklearn not available for k-means fallback.")
                            centroids = np.random.rand(n_clusters, dimension).astype('float32')
        else:
            raise ValueError("Could not extract centroids: CPU index has no quantizer")
    else:
        # Extract from CPU index directly
        if hasattr(index, 'quantizer'):
            if hasattr(index.quantizer, 'centroids'):
                # Direct access to centroids
                centroids = faiss.vector_to_array(index.quantizer.centroids)
                centroids = centroids.reshape(n_clusters, dimension)
            else:
                # Similar logic as above for CPU indexes with complex quantizers
                print("CPU index does not expose centroids directly.")
                centroids = np.zeros((n_clusters, dimension), dtype='float32')
                
                if hasattr(index.quantizer, 'reconstruct'):
                    for i in range(n_clusters):
                        try:
                            index.quantizer.reconstruct(i, centroids[i])
                        except RuntimeError as e:
                            print(f"Warning: Failed to reconstruct centroid {i}: {e}")
                            centroids[i] = np.random.rand(dimension).astype('float32')
                else:
                    print("Warning: No direct method to extract centroids from CPU index.")
                    # Use the same alternative approach as above
                    try:
                        from sklearn.cluster import KMeans
                        if vectors is not None:
                            kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=20).fit(vectors[:min(len(vectors), 100000)])
                            centroids = kmeans.cluster_centers_.astype('float32')
                        else:
                            print("Error: No vectors available for k-means fallback.")
                            centroids = np.random.rand(n_clusters, dimension).astype('float32')
                    except ImportError:
                        print("Error: sklearn not available for k-means fallback.")
                        centroids = np.random.rand(n_clusters, dimension).astype('float32')
        else:
            raise ValueError("Could not extract centroids: Index has no quantizer")
    
    print(f"Extracted codebook with {centroids.shape[0]} centroids")
    return centroids


def get_vector_assignments(index, vectors, config):
    """Get cluster assignments for all vectors."""
    print("Computing cluster assignments for all vectors...")
    
    # For GPU indexes, we may need to handle the quantizer differently
    if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
        # For GPU, we'll create a CPU version of the quantizer for assignment
        cpu_index = faiss.index_gpu_to_cpu(index)
        quantizer = cpu_index.quantizer
    else:
        quantizer = index.quantizer
    
    # Get assignments with progress bar
    batch_size = 10000  # Process in batches to show progress
    num_batches = (vectors.shape[0] + batch_size - 1) // batch_size
    
    assignments = np.zeros(vectors.shape[0], dtype=np.int32)
    
    with tqdm(total=vectors.shape[0], desc="Computing assignments") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, vectors.shape[0])
            batch = vectors[start_idx:end_idx]
            
            _, batch_assignments = quantizer.search(batch, 1)
            assignments[start_idx:end_idx] = batch_assignments.ravel()
            
            pbar.update(batch.shape[0])
    
    # Count vectors per cluster
    unique_clusters, counts = np.unique(assignments, return_counts=True)
    print(f"Vectors are assigned to {len(unique_clusters)} different clusters")
    
    # Calculate some statistics
    min_count = counts.min()
    max_count = counts.max()
    avg_count = counts.mean()
    
    print(f"Cluster sizes - Min: {min_count}, Max: {max_count}, Avg: {avg_count:.2f}")
    
    # Visualize cluster distribution (if not too many clusters)
    if len(unique_clusters) <= 100:  # Only show visualization for a reasonable number of clusters
        plt.figure(figsize=(12, 6))
        plt.bar(unique_clusters, counts)
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Vectors")
        plt.title("Distribution of Vectors across Clusters")
        
        # Save the figure if output directory is configured
        if config['output']['output_dir']:
            output_dir = config['output']['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "cluster_distribution.png"))
            plt.close()
        else:
            plt.show()
    
    return assignments


def save_results(index, codebook, assignments, vector_ids, config):
    """Save the index, codebook, assignments, and vector IDs."""
    if not config['output']['output_dir']:
        print("No output directory specified, skipping save.")
        return
    
    output_dir = config['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save codebook if requested
    if config['output']['save_codebook'] and codebook is not None:
        codebook_path = os.path.join(output_dir, config['output']['codebook_file'])
        print(f"Saving codebook to {codebook_path}")
        np.save(codebook_path, codebook)
    
    # Save index if requested
    if config['output']['save_index']:
        index_path = os.path.join(output_dir, config['output']['index_file'])
        print(f"Saving FAISS index to {index_path}")
        
        # Check if this is a GPU index and convert to CPU if needed
        if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
            print("Converting GPU index to CPU before saving...")
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, index_path)
            except Exception as e:
                print(f"Error converting GPU index to CPU: {e}")
                print("Trying alternative conversion...")
                try:
                    # Another way to convert - create new CPU index and copy parameters
                    cpu_index = faiss.IndexIVFFlat(faiss.IndexFlatL2(index.d), 
                                                index.d, 
                                                index.nlist,
                                                faiss.METRIC_L2)
                    faiss.write_index(cpu_index, index_path)
                except Exception as e:
                    print(f"Failed to save index: {e}")
                    print("Skipping index save. Codebook is still saved and can be used separately.")
        else:
            try:
                faiss.write_index(index, index_path)
            except Exception as e:
                print(f"Failed to save CPU index: {e}")
                print("Skipping index save. Codebook is still saved and can be used separately.")
    
    # Save assignments if requested
    if config['output']['save_assignments'] and assignments is not None:
        assignments_path = os.path.join(output_dir, config['output']['assignments_file'])
        print(f"Saving cluster assignments to {assignments_path}")
        np.save(assignments_path, assignments)
        
        # Also save as CSV with IDs if available
        if vector_ids is not None:
            assignments_csv_path = os.path.join(output_dir, "assignments.csv")
            assignments_df = pd.DataFrame({
                'id': vector_ids,
                'cluster': assignments
            })
            assignments_df.to_csv(assignments_csv_path, index=False)
            print(f"Saved assignments with IDs to {assignments_csv_path}")
    
    print(f"All requested results saved to {output_dir}")


def run_example_queries(index, vectors, codebook, vector_ids, config):
    """Run some example queries to demonstrate the index functionality."""
    if not config['query']['run_query_examples']:
        return
    
    n_queries = config['query']['n_query_examples']
    k = config['query']['k']
    
    print(f"\nRunning {n_queries} example queries...")
    
    # Set nprobe (number of clusters to search)
    index.nprobe = config['clustering']['nprobe']
    
    # Randomly select some vectors as queries
    query_indices = np.random.choice(vectors.shape[0], n_queries, replace=False)
    
    for i, idx in enumerate(query_indices):
        query = vectors[idx:idx+1]  # Keep 2D shape for FAISS
        
        if vector_ids is not None:
            print(f"\nQuery {i+1} (vector ID: {vector_ids[idx]}):")
        else:
            print(f"\nQuery {i+1} (vector index {idx}):")
        
        # Find which cluster the query belongs to
        _, cluster_assignment = index.quantizer.search(query, 1)
        cluster_id = cluster_assignment[0][0]
        
        print(f"  Belongs to cluster: {cluster_id}")
        print(f"  Cluster centroid distance: {np.linalg.norm(query[0] - codebook[cluster_id]):.4f}")
        
        # Search for nearest neighbors
        distances, indices = index.search(query, k)
        
        print(f"  Top {k} nearest vectors:")
        for j, (distance, vector_idx) in enumerate(zip(distances[0], indices[0])):
            if vector_ids is not None and vector_idx < len(vector_ids):
                print(f"    {j+1}. Vector {vector_ids[vector_idx]} - Distance: {distance:.4f}")
            else:
                print(f"    {j+1}. Vector index {vector_idx} - Distance: {distance:.4f}")


def main():
    """Main function to run the FAISS clustering."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FAISS Vector Clustering Tool")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load or generate vector data using the modular loader
    vectors, vector_ids = load_data_from_config(config)
    
    # Create and train FAISS index
    index = create_faiss_index(vectors, config)
    
    # Extract the codebook (centroids)
    # Pass vectors to make them available for fallback method
    codebook = extract_codebook(index, config, vectors)
    
    # Get cluster assignments for all vectors
    assignments = get_vector_assignments(index, vectors, config)
    
    # Add vectors to the index for search
    add_vectors_to_index(index, vectors)
    
    # Save results
    save_results(index, codebook, assignments, vector_ids, config)
    
    # Run example queries
    run_example_queries(index, vectors, codebook, vector_ids, config)
    
    print("\nClustering process completed!")


if __name__ == "__main__":
    main()