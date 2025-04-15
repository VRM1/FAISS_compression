"""
Functions for loading embeddings from parquet files.
"""

import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm


def detect_embedding_columns(df, config=None):
    """
    Detect embedding columns in a DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing embedding columns
        config (dict, optional): Configuration with embedding_columns specification
        
    Returns:
        list: List of embedding column names
    """
    # Check if embedding columns are specified in config
    if config and 'embedding_columns' in config.get('data', {}) and config['data']['embedding_columns']:
        embedding_columns = config['data']['embedding_columns']
        print(f"Using {len(embedding_columns)} embedding columns from config")
        return embedding_columns
    
    # Look for columns that might be embeddings (embed0, embed1, etc.)
    embedding_columns = [col for col in df.columns if col.startswith('embed')]
    
    if embedding_columns:
        print(f"Auto-detected {len(embedding_columns)} embedding columns with 'embed' prefix")
        return embedding_columns
    
    # If no 'embed' columns, try columns with 'vector' in the name
    vector_columns = [col for col in df.columns if 'vector' in col.lower()]
    if vector_columns:
        print(f"Auto-detected {len(vector_columns)} embedding columns with 'vector' in name")
        return vector_columns
    
    # If neither, assume all columns except 'id' and obvious non-embedding columns are embeddings
    non_embedding_cols = ['id', 'user_id', 'item_id', 'label', 'class', 'category', 'timestamp', 'date']
    potential_emb_cols = [col for col in df.columns if col not in non_embedding_cols]
    
    if potential_emb_cols:
        print(f"Auto-detected {len(potential_emb_cols)} potential embedding columns by excluding known non-embedding columns")
        return potential_emb_cols
    
    raise ValueError("Could not detect embedding columns in the DataFrame")


def load_embeddings_from_parquet(file_path, config=None):
    """
    Load vector embeddings from a parquet file.
    
    Args:
        file_path (str): Path to the parquet file containing embeddings
        config (dict, optional): Configuration dictionary
        
    Returns:
        tuple: (numpy.ndarray, list) - Array of embeddings and list of embedding column names
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If embeddings column cannot be found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found at {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Detect embedding columns
    embedding_columns = detect_embedding_columns(df, config)
    
    # Extract embeddings
    embeddings = df[embedding_columns].values.astype('float32')
    
    # Get IDs if available
    ids = None
    if 'id' in df.columns:
        ids = df['id'].tolist()
    
    return embeddings, embedding_columns, ids


def load_metadata_from_directory(directory_path):
    """
    Load metadata file from a directory if it exists.
    
    Args:
        directory_path (str): Directory that might contain a metadata file
        
    Returns:
        dict or None: Metadata dictionary or None if not found
    """
    metadata_path = os.path.join(directory_path, 'metadata.parquet')
    
    if os.path.exists(metadata_path):
        try:
            metadata_df = pd.read_parquet(metadata_path)
            # Convert DataFrame to dict for first row
            metadata = metadata_df.iloc[0].to_dict()
            print(f"Loaded metadata from {metadata_path}")
            return metadata
        except Exception as e:
            print(f"Warning: Could not load metadata file: {e}")
    
    return None


def load_embeddings_from_multiple_parquets(directory_path, config=None, batch_size=None, exclude_files=None):
    """
    Load embeddings from multiple parquet files in a directory.
    
    Args:
        directory_path (str): Directory containing parquet files
        config (dict, optional): Configuration dictionary
        batch_size (int, optional): Size of batches to load (for memory efficiency)
        exclude_files (list, optional): List of filenames to exclude
        
    Returns:
        tuple: (numpy.ndarray, list, list or None) - Combined array of embeddings, 
               list of embedding column names, and list of IDs if available
        
    Raises:
        FileNotFoundError: If directory does not exist or no matching files found
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    # Default exclude
    if exclude_files is None:
        exclude_files = ['metadata.parquet']
    
    # Find all parquet files (excluding specified files)
    parquet_files = [f for f in os.listdir(directory_path) 
                    if f.endswith('.parquet') and f not in exclude_files]
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {directory_path}")
    
    print(f"Found {len(parquet_files)} parquet files to load")
    
    # Try to load metadata if available
    metadata = load_metadata_from_directory(directory_path)
    
    # If metadata has embedding columns, use them
    embedding_columns = None
    if metadata and 'embedding_columns' in metadata:
        embedding_columns = metadata['embedding_columns']
        print(f"Using embedding columns from metadata: {embedding_columns[:5]}... (total: {len(embedding_columns)})")
    elif metadata and 'n_dimensions' in metadata:
        n_dim = metadata['n_dimensions']
        embedding_columns = [f'embed{i}' for i in range(n_dim)]
        print(f"Using {n_dim} embedding columns based on metadata dimensions")
    
    # If batch_size not specified, get it from config or default
    if batch_size is None and config and 'batch_size' in config.get('data', {}):
        batch_size = config['data']['batch_size']
    elif batch_size is None:
        batch_size = 100000  # Default batch size
    
    # Load the first file to determine structure
    first_file = os.path.join(directory_path, parquet_files[0])
    print(f"Loading first file to determine structure: {first_file}")
    
    first_batch, detected_columns, first_ids = load_embeddings_from_parquet(first_file, config)
    
    # Use detected columns if not specified in metadata
    if embedding_columns is None:
        embedding_columns = detected_columns
    
    n_dimensions = len(embedding_columns)
    
    # Count total vectors
    total_vectors = 0
    for file in parquet_files:
        file_path = os.path.join(directory_path, file)
        # Get number of rows without loading the whole file
        df_sample = pd.read_parquet(file_path, columns=['id'] if 'id' in pd.read_parquet(file_path, columns=[]).columns else None)
        total_vectors += len(df_sample)
    
    print(f"Total vectors to load: {total_vectors}")
    
    # Initialize result arrays
    all_embeddings = np.zeros((total_vectors, n_dimensions), dtype='float32')
    all_ids = [] if first_ids is not None else None
    
    # Load first batch
    start_idx = 0
    end_idx = len(first_batch)
    all_embeddings[start_idx:end_idx] = first_batch
    
    if all_ids is not None:
        all_ids.extend(first_ids)
    
    # Load remaining files in batches
    with tqdm(total=total_vectors, desc="Loading vectors") as pbar:
        # Update for first batch
        pbar.update(len(first_batch))
        
        # Process remaining files
        for file in parquet_files[1:]:
            file_path = os.path.join(directory_path, file)
            batch_embeddings, _, batch_ids = load_embeddings_from_parquet(file_path, config)
            
            # Add to the result array
            start_idx = end_idx
            end_idx = start_idx + len(batch_embeddings)
            all_embeddings[start_idx:end_idx] = batch_embeddings
            
            if all_ids is not None and batch_ids is not None:
                all_ids.extend(batch_ids)
            
            pbar.update(len(batch_embeddings))
    
    print(f"Loaded {end_idx} vectors with {n_dimensions} dimensions")
    
    return all_embeddings, embedding_columns, all_ids


def load_data_from_config(config):
    """
    Load data based on configuration, handling both synthetic and parquet data.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (numpy.ndarray, list or None) - Array of embeddings and list of IDs if available
    """
    # Generate synthetic data if specified
    if config['data']['use_synthetic']:
        print(f"Generating {config['data']['n_vectors']} synthetic vectors with {config['data']['n_dimensions']} dimensions...")
        np.random.seed(config['training']['seed'])
        vectors = np.random.random((config['data']['n_vectors'], config['data']['n_dimensions'])).astype('float32')
        ids = [f"synthetic_{i}" for i in range(config['data']['n_vectors'])]
    
    # Load from file or directory
    elif config['data']['data_path']:
        data_path = config['data']['data_path']
        
        # Handle directory of parquet files
        if os.path.isdir(data_path):
            vectors, _, ids = load_embeddings_from_multiple_parquets(
                data_path, 
                config=config,
                batch_size=config['data'].get('batch_size')
            )
        
        # Handle single parquet file
        elif data_path.endswith('.parquet'):
            vectors, _, ids = load_embeddings_from_parquet(data_path, config)
        
        # Handle numpy file
        elif data_path.endswith('.npy'):
            vectors = np.load(data_path).astype('float32')
            ids = [f"numpy_{i}" for i in range(len(vectors))]
        
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
    
    else:
        raise ValueError("Either use_synthetic must be true or data_path must be provided")
    
    # Normalize vectors if requested
    if config['data']['normalize_vectors']:
        print("Normalizing vectors...")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        non_zero_norm = norms > 0
        vectors[non_zero_norm.flatten()] = vectors[non_zero_norm.flatten()] / norms[non_zero_norm]
    
    print(f"Loaded {vectors.shape[0]} vectors with {vectors.shape[1]} dimensions")
    return vectors.astype('float32'), ids  # Ensure float32 format for FAISS