"""
Script to generate synthetic embedding data and save as parquet files.
Creates a dataset with IDs and embedding vectors for testing FAISS clustering.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_batch(start_idx, batch_size, n_dimensions, normalize=True, seed=None):
    """
    Generate a batch of synthetic embedding vectors.
    
    Args:
        start_idx (int): Starting index for IDs
        batch_size (int): Number of vectors to generate
        n_dimensions (int): Dimensionality of embeddings
        normalize (bool): Whether to normalize vectors to unit length
        seed (int, optional): Random seed
        
    Returns:
        pandas.DataFrame: DataFrame with ID and embedding columns
    """
    if seed is not None:
        np.random.seed(seed + start_idx)  # Use different seed for each batch
    
    # Generate random vectors
    vectors = np.random.random((batch_size, n_dimensions)).astype('float32')
    
    # Normalize if requested
    if normalize:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / norms
    
    # Create DataFrame
    df = pd.DataFrame()
    
    # Add ID column
    df['id'] = [f"user_{i}" for i in range(start_idx, start_idx + batch_size)]
    
    # Add embedding columns
    for i in range(n_dimensions):
        df[f'embed{i}'] = vectors[:, i]
    
    return df


def main():
    """Main function to generate synthetic data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Synthetic Embedding Data Generator")
    parser.add_argument("--config", type=str, default="config_data.yml", help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration values
    n_vectors = config['data']['n_vectors']
    n_dimensions = config['data']['n_dimensions']
    batch_size = config['data']['batch_size']
    normalize = config['data']['normalize_vectors']
    seed = config['data']['seed']
    
    output_dir = config['output']['output_dir']
    file_prefix = config['output']['file_prefix']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate number of batches
    n_batches = (n_vectors + batch_size - 1) // batch_size
    
    print(f"Generating {n_vectors} synthetic vectors with {n_dimensions} dimensions")
    print(f"Saving in {n_batches} batches to {output_dir}")
    
    # Generate and save data in batches
    for batch_idx in tqdm(range(n_batches), desc="Generating batches"):
        # Calculate start index and actual batch size for this batch
        start_idx = batch_idx * batch_size
        actual_batch_size = min(batch_size, n_vectors - start_idx)
        
        if actual_batch_size <= 0:
            break
        
        # Generate batch
        batch_seed = seed + batch_idx if seed is not None else None
        df = generate_batch(start_idx, actual_batch_size, n_dimensions, normalize, batch_seed)
        
        # Save to parquet
        output_path = os.path.join(output_dir, f"{file_prefix}_{batch_idx}.parquet")
        df.to_parquet(output_path, index=False)
        
        # Update progress
        vectors_generated = min((batch_idx + 1) * batch_size, n_vectors)
        print(f"Generated {vectors_generated}/{n_vectors} vectors - Saved batch to {output_path}")
    
    print(f"Data generation complete! Generated {n_vectors} vectors across {n_batches} files.")
    
    # Create a metadata file with information about the dataset
    metadata = {
        'n_vectors': n_vectors,
        'n_dimensions': n_dimensions,
        'n_batches': n_batches,
        'normalized': normalize,
        'embedding_columns': [f'embed{i}' for i in range(n_dimensions)]
    }
    
    metadata_df = pd.DataFrame([metadata])
    metadata_path = os.path.join(output_dir, "metadata.parquet")
    metadata_df.to_parquet(metadata_path, index=False)
    
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()