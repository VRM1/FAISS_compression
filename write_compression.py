"""
write_compression.py - Reconstruct compressed vectors and write to CSV

Creates CSV files with reconstructed vectors from FAISS compression:
- Loads codebook (centroids or PQ codebooks)
- Loads assignments (cluster IDs or PQ codes)
- Reconstructs approximate vectors
- Writes to CSV format: ["row_id", "emb1", "emb2", ..., "embN"]

Supports both IVF and PQ compression methods.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
import pdb
from dataset.parquet_loader import load_data_from_config


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_compression_method(codebook_path, assignments_path):
    """Auto-detect compression method from file paths."""
    if 'clustering_PQ' in codebook_path or 'clustering_PQ' in assignments_path:
        return 'PQ'
    elif 'clustering_IVF' in codebook_path or 'clustering_IVF' in assignments_path:
        return 'IVF'
    else:
        raise ValueError("Cannot auto-detect compression method from paths. Ensure paths contain 'clustering_PQ' or 'clustering_IVF'")


def reconstruct_ivf_vectors(codebook, assignments, config):
    """Reconstruct vectors from IVF codebook and assignments."""
    print("Reconstructing IVF vectors...")
    
    n_vectors = len(assignments)
    n_dimensions = codebook.shape[1]
    
    # Reconstruct vectors by looking up centroids
    reconstructed_vectors = np.zeros((n_vectors, n_dimensions), dtype=np.float32)
    
    with tqdm(total=n_vectors, desc="Reconstructing IVF vectors") as pbar:
        for i in range(n_vectors):
            cluster_id = assignments[i]
            reconstructed_vectors[i] = codebook[cluster_id]
            pbar.update(1)
    
    print(f"Reconstructed {n_vectors} vectors with {n_dimensions} dimensions using IVF")
    return reconstructed_vectors


def reconstruct_pq_vectors(codebook, assignments, config):
    """Reconstruct vectors from PQ codebook and assignments."""
    print("Reconstructing PQ vectors...")
    
    n_vectors, code_size = assignments.shape
    
    # Auto-detect PQ parameters from codebook shape
    m = codebook.shape[0]  # Number of sub-quantizers
    n_centroids = codebook.shape[1]  # Centroids per sub-quantizer  
    sub_dim = codebook.shape[2]  # Dimensions per sub-vector
    
    # Calculate bits from number of centroids
    bits = int(np.log2(n_centroids))
    n_dimensions = m * sub_dim  # Total dimensions
    
    print(f"Auto-detected PQ parameters: {m} sub-quantizers, {bits} bits each")
    print(f"Codebook shape: {codebook.shape}")
    print(f"Total dimensions: {n_dimensions}")
    
    # Verify our detection matches the code size
    expected_code_size = (m * bits + 7) // 8  # Round up to bytes
    if code_size != expected_code_size:
        print(f"Warning: Expected code size {expected_code_size}, got {code_size}")
    
    # Reconstruct vectors using PQ codebooks
    reconstructed_vectors = np.zeros((n_vectors, n_dimensions), dtype=np.float32)
    
    # Create temporary FAISS index with CORRECT auto-detected parameters
    temp_index = faiss.IndexPQ(n_dimensions, m, bits)
    
    # Set the PQ centroids
    pq_centroids_flat = codebook.reshape(-1)
    faiss.copy_array_to_vector(pq_centroids_flat, temp_index.pq.centroids)
    temp_index.is_trained = True
    
    # Decode in batches for efficiency
    batch_size = 1000
    num_batches = (n_vectors + batch_size - 1) // batch_size
    
    with tqdm(total=n_vectors, desc="Reconstructing PQ vectors") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_vectors)
            
            batch_codes = assignments[start_idx:end_idx]
            batch_size_actual = end_idx - start_idx
            
            # Reconstruct this batch
            batch_reconstructed = np.zeros((batch_size_actual, n_dimensions), dtype=np.float32)
            temp_index.sa_decode(batch_codes, batch_reconstructed)
            
            reconstructed_vectors[start_idx:end_idx] = batch_reconstructed
            pbar.update(batch_size_actual)
    
    print(f"Reconstructed {n_vectors} vectors with {n_dimensions} dimensions using PQ")
    return reconstructed_vectors


def create_embedding_headers(n_dimensions):
    """Create column headers for embedding CSV."""
    headers = ["row_id"]
    headers.extend([f"emb{i+1}" for i in range(n_dimensions)])
    return headers


def write_reconstructed_vectors_to_csv(vectors, output_path, config, method, batch_size=10000):
    """Write reconstructed vectors to CSV file with optional error calculation."""
    n_vectors, n_dimensions = vectors.shape
    
    print(f"Writing {n_vectors} reconstructed vectors with {n_dimensions} dimensions to {output_path}")
    
    # Load true embeddings if error calculation is enabled
    error_enabled = config.get('error_calculation', {}).get('enabled', False)
    true_vectors = None
    all_errors = []
    
    if error_enabled:
        try:
            true_vectors = load_true_embeddings(config)
            print("Error calculation enabled - will sample vectors during CSV creation")
        except Exception as e:
            print(f"Warning: Could not load true embeddings for error calculation: {e}")
            error_enabled = False
    
    # Create headers
    headers = create_embedding_headers(n_dimensions)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write CSV in batches
    num_batches = (n_vectors + batch_size - 1) // batch_size
    
    with tqdm(total=n_vectors, desc="Writing CSV") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_vectors)
            
            # Get batch data
            batch_vectors = vectors[start_idx:end_idx]
            batch_ids = list(range(start_idx, end_idx))
            
            # Calculate errors for this batch if enabled
            if error_enabled and true_vectors is not None:
                batch_errors = calculate_batch_errors(batch_vectors, true_vectors, start_idx, config)
                all_errors.extend(batch_errors)
                
                # Limit total samples
                max_samples = config.get('error_calculation', {}).get('total_samples', 1000)
                if len(all_errors) >= max_samples:
                    print(f"Reached maximum sample limit ({max_samples}), stopping error collection")
                    error_enabled = False
            
            # Create DataFrame for this batch
            batch_data = {'row_id': batch_ids}
            
            # Add embedding columns
            for i in range(n_dimensions):
                batch_data[f'emb{i+1}'] = batch_vectors[:, i]
            
            batch_df = pd.DataFrame(batch_data)
            
            # Write to CSV (append after first batch)
            mode = 'w' if batch_idx == 0 else 'a'
            header = batch_idx == 0
            
            batch_df.to_csv(output_path, mode=mode, header=header, index=False)
            
            pbar.update(len(batch_vectors))
    
    print(f"Successfully wrote reconstructed vectors to {output_path}")
    
    # Report error statistics if we collected any
    if all_errors:
        error_stats = report_error_statistics(all_errors, method)
        return output_path, error_stats
    
    return output_path, None


def calculate_compression_stats(codebook, assignments, method):
    """Calculate compression statistics."""
    print(f"\n=== COMPRESSION STATISTICS ({method}) ===")
    
    if method == 'IVF':
        n_vectors = len(assignments)
        n_dimensions = codebook.shape[1]
        n_clusters = codebook.shape[0]
        
        # Original size: n_vectors * n_dimensions * 4 bytes (float32)
        original_size = n_vectors * n_dimensions * 4
        
        # Compressed size: codebook + assignments
        codebook_size = codebook.nbytes
        assignments_size = assignments.nbytes
        compressed_size = codebook_size + assignments_size
        
        print(f"Vectors: {n_vectors:,}")
        print(f"Dimensions: {n_dimensions}")
        print(f"Clusters: {n_clusters}")
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Codebook size: {codebook_size / 1024 / 1024:.2f} MB")
        print(f"Assignments size: {assignments_size / 1024 / 1024:.2f} MB")
        print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {original_size / compressed_size:.2f}x")
        print(f"Space savings: {(1 - compressed_size/original_size) * 100:.1f}%")
        
    elif method == 'PQ':
        n_vectors, code_size = assignments.shape
        n_dimensions = codebook.shape[0] * codebook.shape[2]  # m * (d/m)
        
        # Original size
        original_size = n_vectors * n_dimensions * 4
        
        # Compressed size
        codebook_size = codebook.nbytes
        assignments_size = assignments.nbytes
        compressed_size = codebook_size + assignments_size
        
        print(f"Vectors: {n_vectors:,}")
        print(f"Dimensions: {n_dimensions}")
        print(f"Sub-quantizers: {codebook.shape[0]}")
        print(f"Centroids per sub-quantizer: {codebook.shape[1]}")
        print(f"Code size per vector: {code_size} bytes")
        print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"Codebook size: {codebook_size / 1024 / 1024:.2f} MB")
        print(f"Assignments size: {assignments_size / 1024 / 1024:.2f} MB")
        print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"Compression ratio: {original_size / compressed_size:.2f}x")
        print(f"Space savings: {(1 - compressed_size/original_size) * 100:.1f}%")


def preview_csv(csv_path, n_rows=5):
    """Preview the created CSV file."""
    print(f"\n=== PREVIEW: {csv_path} ===")
    
    # Get file size
    file_size = os.path.getsize(csv_path) / 1024 / 1024
    print(f"File size: {file_size:.2f} MB")
    
    # Read first few rows
    df = pd.read_csv(csv_path, nrows=n_rows)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Show some statistics
    embedding_cols = [col for col in df.columns if col.startswith('emb')]
    if embedding_cols:
        print(f"\nEmbedding statistics (sample):")
        print(df[embedding_cols].describe())


def main():
    """Main function to reconstruct compressed vectors and write to CSV."""
    parser = argparse.ArgumentParser(description="Reconstruct compressed vectors and write to CSV")
    parser.add_argument("--config", type=str, default="write_compression_config.yml", 
                       help="Path to configuration file")
    parser.add_argument("--codebook", type=str, help="Path to codebook file (overrides config)")
    parser.add_argument("--assignments", type=str, help="Path to assignments file (overrides config)")
    parser.add_argument("--output", type=str, help="Output CSV file path (overrides config)")
    parser.add_argument("--method", choices=['IVF', 'PQ'], help="Compression method (auto-detected if not specified)")
    parser.add_argument("--preview", action="store_true", help="Preview the output CSV")
    parser.add_argument("--stats", action="store_true", help="Show compression statistics")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found. Using default settings.")
        config = {
            'paths': {
                'codebook': 'results/clustering_IVF/codebook.npy',
                'assignments': 'results/clustering_IVF/assignments.npy'
            },
            'output': {
                'csv_path': 'output/reconstructed_embeddings.csv',
                'batch_size': 10000
            },
            'clustering': {
                'pq_m': 8,
                'pq_bits': 8
            },
            'data': {
                'n_dimensions': 200
            }
        }
    
    # Override paths if provided
    codebook_path = args.codebook or config['paths']['codebook']
    assignments_path = args.assignments or config['paths']['assignments']
    output_path = args.output or config['output']['csv_path']
    
    print("=== WRITE COMPRESSION TOOL ===")
    print(f"Codebook: {codebook_path}")
    print(f"Assignments: {assignments_path}")
    print(f"Output: {output_path}")
    
    # Verify files exist
    if not os.path.exists(codebook_path):
        raise FileNotFoundError(f"Codebook file not found: {codebook_path}")
    if not os.path.exists(assignments_path):
        raise FileNotFoundError(f"Assignments file not found: {assignments_path}")
    
    # Detect compression method
    method = args.method or detect_compression_method(codebook_path, assignments_path)
    print(f"Compression method: {method}")
    
    # Load data
    print(f"Loading codebook from {codebook_path}")
    codebook = np.load(codebook_path)
    print(f"Codebook shape: {codebook.shape}")
    
    print(f"Loading assignments from {assignments_path}")
    assignments = np.load(assignments_path)
    print(f"Assignments shape: {assignments.shape}")
    # Read config settings (combine with command line args)
    show_stats = args.stats or config['output'].get('show_stats', False)
    preview_csv_flag = args.preview or config['output'].get('preview_csv', False)
    preview_rows = config['output'].get('preview_rows', 5)

    # Show compression statistics if requested
    if show_stats:
        calculate_compression_stats(codebook, assignments, method)

    # Reconstruct vectors
    if method == 'IVF':
        reconstructed_vectors = reconstruct_ivf_vectors(codebook, assignments, config)
    elif method == 'PQ':
        reconstructed_vectors = reconstruct_pq_vectors(codebook, assignments, config)
    else:
        raise ValueError(f"Unknown compression method: {method}")

    # Write to CSV
    batch_size = config['output'].get('batch_size', 10000)
    output_path, error_stats = write_reconstructed_vectors_to_csv(reconstructed_vectors, output_path, config, method, batch_size)


    # Preview if requested
    if preview_csv_flag:
        preview_csv(output_path, preview_rows)
    
    print(f"\n✅ Successfully created CSV: {output_path}")
    print(f"📊 Contains {reconstructed_vectors.shape[0]} reconstructed vectors")
    print(f"📏 Each vector has {reconstructed_vectors.shape[1]} dimensions")
    print(f"🗜️  Reconstructed using {method} compression")
    if error_stats:
        print(f"📏 Reconstruction quality: Mean error = {error_stats['mean_error']:.6f}")


def load_true_embeddings(config):
    """Load true embeddings for error calculation."""
    print("Loading true embeddings for error calculation...")
    
    # Create temporary config for loading data
    temp_config = {
        'data': config['data']
    }
    
    vectors, vector_ids = load_data_from_config(temp_config)
    print(f"Loaded {vectors.shape[0]} true vectors with {vectors.shape[1]} dimensions")
    return vectors

def calculate_batch_errors(reconstructed_batch, true_vectors, batch_start_idx, config):
    """Calculate errors for a random sample from the current batch."""
    error_config = config.get('error_calculation', {})
    samples_per_batch = error_config.get('samples_per_batch', 100)
    
    batch_size = len(reconstructed_batch)
    n_samples = min(samples_per_batch, batch_size)
    
    # Randomly sample indices from this batch
    sample_indices = np.random.choice(batch_size, n_samples, replace=False)
    
    errors = []
    for idx in sample_indices:
        true_idx = batch_start_idx + idx
        if true_idx < len(true_vectors):
            reconstructed_vec = reconstructed_batch[idx]
            true_vec = true_vectors[true_idx]
            
            # Calculate L2 error
            error = np.linalg.norm(reconstructed_vec - true_vec)
            errors.append(error)
    
    return errors

def report_error_statistics(all_errors, method):
    """Report final error statistics."""
    if not all_errors:
        print("No error samples collected.")
        return None
    
    errors = np.array(all_errors)
    
    error_stats = {
        'method': method,
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
    
    print(f"\n=== RECONSTRUCTION ERROR RESULTS ({method}) ===")
    print(f"Samples processed: {error_stats['n_samples']}")
    print(f"Mean error: {error_stats['mean_error']:.6f}")
    print(f"Standard deviation: {error_stats['std_error']:.6f}")
    print(f"Min error: {error_stats['min_error']:.6f}")
    print(f"Max error: {error_stats['max_error']:.6f}")
    print(f"Median error: {error_stats['median_error']:.6f}")
    print(f"95th percentile: {error_stats['percentile_95']:.6f}")
    print(f"99th percentile: {error_stats['percentile_99']:.6f}")
    
    return error_stats

if __name__ == "__main__":
    main()