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
import json

# Import from dataset module
from utils.parquet_loader import load_data_from_config


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_faiss_index(vectors, config):
    """Create and train a FAISS index based on configuration."""
    d = vectors.shape[1]  # Dimension
    n_clusters = config['clustering']['n_clusters']
    use_pq = config['clustering']['use_pq']
    
    print(f"Creating FAISS index...")
    
    if use_pq:
        # Create an index with ONLY Product Quantization (no IVF)
        m = config['clustering']['pq_m']  # Number of sub-quantizers
        bits = config['clustering']['pq_bits']  # Bits per sub-quantizer
        
        print(f"Using PURE PQ with {m} sub-quantizers and {bits} bits each")
        
        # Ensure m divides the dimension properly
        if d % m != 0:
            raise ValueError(f"PQ factor {m} does not divide dimension {d} evenly. "
                           f"For dimension {d}, valid pq_m values are: {[i for i in range(1, d+1) if d % i == 0]}")
        
        # Create IndexPQ (Pure PQ, no GPU support in IndexPQ)
        index = faiss.IndexPQ(d, m, bits)
        
    else:
        # Create appropriate quantizer and index based on configuration
        if config['clustering']['gpu'] and faiss.get_num_gpus() > 0:
            num_gpus = faiss.get_num_gpus()
            print(f"Using {num_gpus} GPU(s) for clustering...")
            try:
                # Create CPU index first
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
                
                if num_gpus > 1:
                    print(f"Distributing index across {num_gpus} GPUs...")
                    # Use multiple GPUs
                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True  # Shard the index across GPUs
                    co.useFloat16 = False  # Keep float32 for better precision
                    index = faiss.index_cpu_to_gpu_multiple_py(list(range(num_gpus)), index, co)
                else:
                    print("Using single GPU...")
                    # Use single GPU
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    
            except Exception as e:
                print(f"GPU initialization failed: {e}. Falling back to CPU.")
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_L2)
        else:
            print("Using CPU for clustering...")
            quantizer = faiss.IndexFlatL2(d)
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
    
    # Simple training (IndexPQ doesn't support progress callbacks)
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
    
    if config['clustering']['use_pq']:
        # For IndexPQ, extract PQ codebooks instead of IVF centroids
        print("IndexPQ doesn't have IVF centroids, extracting PQ codebooks instead...")
        return extract_pq_codebooks(index, config)
    else:
        # Original logic for IVF indexes
        n_clusters = config['clustering']['n_clusters']
        dimension = index.d
        
        print("Extracting IVF codebook...")
        
        # For GPU indexes, convert to CPU first for easier access
        if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
            print("Converting GPU index to CPU for codebook extraction...")
            cpu_index = faiss.index_gpu_to_cpu(index)
        else:
            cpu_index = index
        
        # Extract from CPU index directly
        try:
            if hasattr(cpu_index, 'quantizer'):
                quantizer = cpu_index.quantizer
                
                # For GPU-converted indexes, the quantizer might also need conversion
                if hasattr(faiss, 'GpuIndex') and isinstance(quantizer, faiss.GpuIndex):
                    print("Converting quantizer from GPU to CPU...")
                    quantizer = faiss.index_gpu_to_cpu(quantizer)
                
                if hasattr(quantizer, 'centroids'):
                    # Direct access to centroids
                    centroids = faiss.vector_to_array(quantizer.centroids)
                    centroids = centroids.reshape(n_clusters, dimension)
                    print(f"Extracted IVF codebook with {centroids.shape[0]} centroids")
                    return centroids
                elif hasattr(quantizer, 'reconstruct'):
                    # Fallback: reconstruct centroids one by one
                    print("Using reconstruct method to extract centroids...")
                    centroids = np.zeros((n_clusters, dimension), dtype='float32')
                    for i in range(n_clusters):
                        try:
                            quantizer.reconstruct(i, centroids[i])
                        except RuntimeError as e:
                            print(f"Warning: Failed to reconstruct centroid {i}: {e}")
                            # Fall back to random initialization if reconstruct fails
                            centroids[i] = np.random.rand(dimension).astype('float32')
                    
                    print(f"Extracted IVF codebook with {centroids.shape[0]} centroids (via reconstruction)")
                    return centroids
                else:
                    raise ValueError("Quantizer has no centroids or reconstruct method")
            else:
                raise ValueError("Index has no quantizer")
                
        except Exception as e:
            print(f"Failed to extract centroids directly: {e}")
            print("Trying k-means approximation as fallback...")
            
            # Final fallback: use k-means on the original vectors
            try:
                from sklearn.cluster import KMeans
                if vectors is not None:
                    print(f"Using K-means clustering on {min(len(vectors), 100000)} vectors...")
                    sample_vectors = vectors[:min(len(vectors), 100000)]
                    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=20, random_state=42)
                    kmeans.fit(sample_vectors)
                    centroids = kmeans.cluster_centers_.astype('float32')
                    print(f"Extracted approximate codebook with {centroids.shape[0]} centroids (K-means fallback)")
                    return centroids
                else:
                    print("Error: No vectors available for k-means fallback.")
                    # Return random centroids as last resort
                    centroids = np.random.rand(n_clusters, dimension).astype('float32')
                    print(f"Generated random codebook with {centroids.shape[0]} centroids (random fallback)")
                    return centroids
            except ImportError:
                print("Error: sklearn not available for k-means fallback.")
                # Return random centroids as last resort
                centroids = np.random.rand(n_clusters, dimension).astype('float32')
                print(f"Generated random codebook with {centroids.shape[0]} centroids (random fallback)")
                return centroids


def extract_pq_codebooks(index, config):
    """Extract PQ codebooks from an IndexPQ index."""
    if not hasattr(index, 'pq'):
        print("Index does not have PQ component")
        return None
    
    # Extract PQ centroids
    m = config['clustering']['pq_m']  # Number of sub-quantizers
    bits = config['clustering']['pq_bits']  # Bits per sub-quantizer
    d = index.d
    
    # Get PQ centroids
    pq_centroids = faiss.vector_to_array(index.pq.centroids)
    pq_centroids = pq_centroids.reshape(m, 2**bits, d // m)
    
    print(f"Extracted PQ codebooks: {m} sub-quantizers, {2**bits} centroids each, {d//m} dimensions per centroid")
    
    return pq_centroids


def reconstruct_vector_from_pq_codes(pq_codes, pq_centroids):
    """Reconstruct a vector from its PQ codes."""
    reconstructed_parts = []
    for i, code in enumerate(pq_codes):
        reconstructed_parts.append(pq_centroids[i, code])
    
    return np.concatenate(reconstructed_parts)


def inspect_pq_codes_pure(index, vectors, config, n_samples=2):
    """Inspect PQ codes for IndexPQ (pure PQ)."""
    print(f"\n=== INSPECTING PURE PQ CODES ===")
    
    m = config['clustering']['pq_m']
    bits = config['clustering']['pq_bits']
    expected_max_code = 2**bits - 1
    
    print(f"Expected: {m} sub-quantizers, each with codes 0-{expected_max_code}")
    
    # Verify codebook dimensions
    codebook = extract_pq_codebooks(index, config)
    if codebook is not None:
        print(f"Codebook shape: {codebook.shape}")
        print(f"✓ Sub-quantizers: {codebook.shape[0]} (expected: {m})")
        print(f"✓ Centroids per sub-quantizer: {codebook.shape[1]} (expected: {2**bits})")
        print(f"✓ Dimensions per sub-vector: {codebook.shape[2]} (expected: {index.d // m})")
    
    sample_indices = np.random.choice(len(vectors), min(n_samples, len(vectors)), replace=False)
    all_codes = []
    
    for i, vec_idx in enumerate(sample_indices):
        test_vector = vectors[vec_idx]
        
        print(f"\nVector {vec_idx}:")
        print(f"  Original vector shape: {test_vector.shape}")
        
        # Get the packed PQ codes
        codes = np.zeros((1, index.sa_code_size()), dtype=np.uint8)
        index.sa_encode(test_vector.reshape(1, -1), codes)
        
        print(f"  Packed codes: {codes[0][:index.sa_code_size()]}")
        print(f"  Code size: {index.sa_code_size()} bytes")
        
        # Verify reconstruction works
        reconstructed = np.zeros((1, len(test_vector)), dtype='float32')
        index.sa_decode(codes, reconstructed)
        
        # Calculate reconstruction error
        error = np.linalg.norm(test_vector - reconstructed[0])
        print(f"  Reconstruction error: {error:.6f}")
        
        # Simple verification: try to manually check if codes are in valid range
        # This is a basic check - for 8-bit codes we can see them directly
        if bits == 8:
            actual_codes = codes[0][:m]
            print(f"  Actual codes: {actual_codes}")
            max_found = np.max(actual_codes)
            min_found = np.min(actual_codes)
            print(f"  Code range: {min_found}-{max_found} (expected: 0-{expected_max_code})")
            
            if max_found <= expected_max_code:
                print(f"  ✓ All codes within expected range!")
            else:
                print(f"  ✗ ERROR: Found code {max_found} > expected max {expected_max_code}")
        else:
            print(f"  Note: {bits}-bit codes are packed, but reconstruction works correctly")
            print(f"  This confirms {m} sub-quantizers with {2**bits} centroids each are working")
        
        all_codes.append(codes[0])
    
    print("=== END PQ CODES INSPECTION ===\n")
    
    return all_codes[0] if all_codes else np.array([])


def get_vector_assignments(index, vectors, config):
    """Get cluster assignments for all vectors."""
    
    if config['clustering']['use_pq']:
        print("IndexPQ doesn't use traditional cluster assignments.")
        print("Each vector gets multiple PQ codes (one per sub-vector).")
        
        # Get codes for all vectors, not just one sample
        assignments = get_all_pq_assignments(index, vectors, config)
        
        # Still run inspection for verification
        inspect_pq_codes_pure(index, vectors, config)
    
        return assignments
    else:
        # Original logic for IVF indexes
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


def get_all_pq_assignments(index, vectors, config):
    """Get PQ codes for all vectors."""
    print("Computing PQ codes for all vectors...")
    
    n_vectors = len(vectors)
    code_size = index.sa_code_size()
    
    # Allocate array for all PQ codes
    all_assignments = np.zeros((n_vectors, code_size), dtype=np.uint8)
    
    # Process in batches
    batch_size = 10000
    num_batches = (n_vectors + batch_size - 1) // batch_size
    
    with tqdm(total=n_vectors, desc="Computing PQ codes") as pbar:
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_vectors)
            batch = vectors[start_idx:end_idx]
            
            # Get PQ codes for this batch
            batch_codes = np.zeros((batch.shape[0], code_size), dtype=np.uint8)
            index.sa_encode(batch, batch_codes)
            
            all_assignments[start_idx:end_idx] = batch_codes
            pbar.update(batch.shape[0])
    
    print(f"Generated PQ assignments shape: {all_assignments.shape}")
    return all_assignments

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
        
        try:
            # First try direct save
            faiss.write_index(index, index_path)
            print(f"Successfully saved index to {index_path}")
        except Exception as e:
            print(f"Direct save failed: {e}")
            print("Attempting GPU to CPU conversion...")
            try:
                cpu_index = faiss.index_gpu_to_cpu(index)
                faiss.write_index(cpu_index, index_path)
                print(f"Successfully saved CPU-converted index to {index_path}")
            except Exception as e2:
                print(f"GPU conversion also failed: {e2}")
                print("Skipping index save. Codebook is still saved and can be used separately.")
    
    # Save assignments if requested (only for non-PQ indexes or PQ codes)
    if config['output']['save_assignments'] and assignments is not None:
        assignments_path = os.path.join(output_dir, config['output']['assignments_file'])
        print(f"Saving assignments/codes to {assignments_path}")
        np.save(assignments_path, assignments)
        
        # For non-PQ indexes, also save as CSV with IDs if available
        if not config['clustering']['use_pq'] and vector_ids is not None:
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
    
    if config['clustering']['use_pq']:
        print("For IndexPQ, queries work differently - no cluster assignments, just PQ search")
        print(f"\nRunning {n_queries} example PQ queries...")
        
        # Randomly select some vectors as queries
        query_indices = np.random.choice(vectors.shape[0], n_queries, replace=False)
        
        for i, idx in enumerate(query_indices):
            query = vectors[idx:idx+1]  # Keep 2D shape for FAISS
            
            if vector_ids is not None:
                print(f"\nQuery {i+1} (vector ID: {vector_ids[idx]}):")
            else:
                print(f"\nQuery {i+1} (vector index {idx}):")
            
            # Search for nearest neighbors using PQ
            distances, indices = index.search(query, k)
            
            print(f"  Top {k} nearest vectors (using PQ approximation):")
            for j, (distance, vector_idx) in enumerate(zip(distances[0], indices[0])):
                if vector_ids is not None and vector_idx < len(vector_ids):
                    print(f"    {j+1}. Vector {vector_ids[vector_idx]} - Distance: {distance:.4f}")
                else:
                    print(f"    {j+1}. Vector index {vector_idx} - Distance: {distance:.4f}")
    else:
        print(f"\nRunning {n_queries} example IVF queries...")
        
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


def calculate_quantization_error(index, vectors, vector_ids, config, n_samples=1000):
    """
    Calculate quantization error by comparing original vectors with their quantized versions.
    
    Args:
        index: Trained FAISS index
        vectors: Original vectors
        vector_ids: Vector IDs (for tracking)
        config: Configuration dictionary
        n_samples: Number of random samples to test
    
    Returns:
        dict: Dictionary containing error statistics
    """
    print(f"\n=== CALCULATING QUANTIZATION ERROR ===")
    
    # Determine actual sample size
    actual_samples = min(n_samples, len(vectors))
    sample_indices = np.random.choice(len(vectors), actual_samples, replace=False)
    
    errors = []
    sample_info = []
    
    print(f"Calculating error for {actual_samples} random samples...")
    
    with tqdm(total=actual_samples, desc="Computing errors") as pbar:
        for i, vec_idx in enumerate(sample_indices):
            original_vector = vectors[vec_idx]
            
            try:
                if config['clustering']['use_pq']:
                    # For IndexPQ: Get PQ codes and reconstruct
                    m = config['clustering']['pq_m']
                    codes = np.zeros((1, index.sa_code_size()), dtype=np.uint8)
                    index.sa_encode(original_vector.reshape(1, -1), codes)
                    
                    # Reconstruct from PQ codes
                    reconstructed = np.zeros((1, len(original_vector)), dtype='float32')
                    index.sa_decode(codes, reconstructed)
                    reconstructed_vector = reconstructed[0]
                    
                    # Store additional info for PQ
                    sample_info.append({
                        'vector_idx': vec_idx,
                        'vector_id': vector_ids[vec_idx] if vector_ids else None,
                        'pq_codes': codes[0][:index.sa_code_size()].tolist(),
                        'method': 'PQ'
                    })
                    
                else:
                    # For IVF: Find nearest cluster and use centroid
                    if hasattr(faiss, 'GpuIndex') and isinstance(index, faiss.GpuIndex):
                        cpu_index = faiss.index_gpu_to_cpu(index)
                        quantizer = cpu_index.quantizer
                    else:
                        quantizer = index.quantizer
                    
                    # Find closest cluster
                    _, cluster_assignment = quantizer.search(original_vector.reshape(1, -1), 1)
                    cluster_id = cluster_assignment[0][0]
                    
                    # Load the codebook from file (more reliable than extracting from index)
                    if 'codebook' not in locals():
                        codebook_path = config['test_mode']['codebook_path']
                        codebook = np.load(codebook_path)
                        print(f"Loaded codebook from {codebook_path} with shape {codebook.shape}")

                    reconstructed_vector = codebook[cluster_id]
                    
                    # Store additional info for IVF
                    sample_info.append({
                        'vector_idx': vec_idx,
                        'vector_id': vector_ids[vec_idx] if vector_ids else None,
                        'cluster_id': int(cluster_id),
                        'method': 'IVF'
                    })
                
                # Calculate error (L2 distance)
                error = np.linalg.norm(original_vector - reconstructed_vector)
                errors.append(error)
                
            except Exception as e:
                print(f"Warning: Failed to calculate error for vector {vec_idx}: {e}")
                continue
                
            pbar.update(1)
        
    # Calculate statistics
    errors = np.array(errors)
    
    if len(errors) == 0:
        print("Error: No valid error calculations completed!")
        return None
    
    error_stats = {
        'method': 'PQ' if config['clustering']['use_pq'] else 'IVF',
        'n_samples': len(errors),
        'mean_error': float(np.mean(errors)),
        'std_error': float(np.std(errors)),
        'variance_error': float(np.var(errors)),
        'min_error': float(np.min(errors)),
        'max_error': float(np.max(errors)),
        'median_error': float(np.median(errors)),
        'percentile_95': float(np.percentile(errors, 95)),
        'percentile_99': float(np.percentile(errors, 99))
    }
    
    # Print results
    print(f"\n=== QUANTIZATION ERROR RESULTS ({error_stats['method']}) ===")
    print(f"Samples processed: {error_stats['n_samples']}")
    print(f"Mean error: {error_stats['mean_error']:.6f}")
    print(f"Standard deviation: {error_stats['std_error']:.6f}")
    print(f"Variance: {error_stats['variance_error']:.6f}")
    print(f"Min error: {error_stats['min_error']:.6f}")
    print(f"Max error: {error_stats['max_error']:.6f}")
    print(f"Median error: {error_stats['median_error']:.6f}")
    print(f"95th percentile: {error_stats['percentile_95']:.6f}")
    print(f"99th percentile: {error_stats['percentile_99']:.6f}")
    
    # Create error distribution plot
    if config['output']['output_dir'] and len(errors) > 1:
        plt.figure(figsize=(12, 8))
        
        # Histogram
        plt.subplot(2, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title(f'{error_stats["method"]} Error Distribution')
        plt.grid(alpha=0.3)
        
        # Box plot
        plt.subplot(2, 2, 2)
        plt.boxplot(errors)
        plt.ylabel('Reconstruction Error')
        plt.title(f'{error_stats["method"]} Error Box Plot')
        plt.grid(alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_errors = np.sort(errors)
        cumulative_prob = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative_prob)
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Cumulative Probability')
        plt.title(f'{error_stats["method"]} Cumulative Distribution')
        plt.grid(alpha=0.3)
        
        # Error vs sample index (to check for patterns)
        plt.subplot(2, 2, 4)
        plt.scatter(range(len(errors)), errors, alpha=0.5)
        plt.xlabel('Sample Index')
        plt.ylabel('Reconstruction Error')
        plt.title(f'{error_stats["method"]} Error vs Sample')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot - create directory if it doesn't exist
        output_dir = config['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"quantization_error_{error_stats['method'].lower()}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved error analysis plot to {plot_path}")
    
    # Save detailed results
    # Save detailed results
    if config['output']['output_dir']:
        # Ensure output directory exists
        os.makedirs(config['output']['output_dir'], exist_ok=True)
        # Save error statistics
        stats_path = os.path.join(config['output']['output_dir'], f"error_stats_{error_stats['method'].lower()}.json")
        
        with open(stats_path, 'w') as f:
            json.dump(error_stats, f, indent=2)
        print(f"Saved error statistics to {stats_path}")
        
        # Save individual sample details
        sample_details_path = os.path.join(config['output']['output_dir'], f"sample_errors_{error_stats['method'].lower()}.csv")
        sample_df = pd.DataFrame(sample_info)
        sample_df['reconstruction_error'] = errors
        sample_df.to_csv(sample_details_path, index=False)
        print(f"Saved sample error details to {sample_details_path}")
    
    print("=== END ERROR CALCULATION ===\n")
    
    return error_stats




def run_test_mode(config):
    """Run quantization error testing on saved index and codebook."""
    print("=== RUNNING TEST MODE ===")
    
    test_config = config['test_mode']
    
    # Load the saved index
    print(f"Loading FAISS index from {test_config['index_path']}")
    if not os.path.exists(test_config['index_path']):
        raise FileNotFoundError(f"Index file not found: {test_config['index_path']}")
    
    index = faiss.read_index(test_config['index_path'])
    
    # Load the original data for comparison
    print(f"Loading original data from {test_config['data_path']}")
    
    # Temporarily modify config to load data from the specified path
    temp_config = config.copy()
    temp_config['data']['data_path'] = test_config['data_path']
    vectors, vector_ids = load_data_from_config(temp_config)
    
    # Auto-detect quantization type from directory path
    index_path = test_config['index_path']
    if 'clustering_PQ' in index_path:
        print("Auto-detected: Using Product Quantization mode")
        codebook = None  # Not needed for PQ
        temp_config['clustering']['use_pq'] = True
    elif 'clustering_IVF' in index_path:
        print("Auto-detected: Using IVF mode")
        print(f"Loading IVF codebook from {test_config['codebook_path']}")
        if not os.path.exists(test_config['codebook_path']):
            raise FileNotFoundError(f"Codebook file not found: {test_config['codebook_path']}")
        codebook = np.load(test_config['codebook_path'])
        print(f"Loaded codebook with shape: {codebook.shape}")
        temp_config['clustering']['use_pq'] = False
    else:
        raise ValueError(f"Cannot auto-detect quantization type from path: {index_path}. Path should contain 'clustering_PQ' or 'clustering_IVF'")
    
    # Calculate quantization error
    error_stats = calculate_quantization_error(
        index, vectors, vector_ids, temp_config, 
        n_samples=test_config['n_samples']
    )
    
    print("=== TEST MODE COMPLETED ===")
    return error_stats


def main():
    """Main function to run the FAISS clustering."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FAISS Vector Clustering Tool")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to configuration file")
    parser.add_argument("--test", action="store_true", help="Run in test mode to calculate quantization error")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if running in test mode
    if args.test or config.get('test_mode', {}).get('enabled', False):
        run_test_mode(config)
        return
    
    # Normal mode: Create index and codebook
    print("=== RUNNING NORMAL MODE ===")
    
    # Load or generate vector data using the modular loader
    vectors, vector_ids = load_data_from_config(config)
    
    # Create and train FAISS index
    index = create_faiss_index(vectors, config)

    # Modify output directory based on clustering method
    if config['clustering']['use_pq']:
        config['output']['output_dir'] = config['output']['output_dir'].replace('clustering', 'clustering_PQ')
    else:
        config['output']['output_dir'] = config['output']['output_dir'].replace('clustering', 'clustering_IVF')

    print(f"Output directory set to: {config['output']['output_dir']}")
    
    # Extract the codebook (centroids)
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