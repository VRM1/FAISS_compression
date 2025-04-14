# FAISS Vector Search Project

A modular implementation of vector similarity search using FAISS for efficient nearest neighbor retrieval.

## Overview

This project provides a modular framework for experimenting with different vector similarity search methods, including:

1. Brute force cosine similarity search
2. FAISS flat L2 search (exact nearest neighbors)
3. FAISS IVF search (approximate nearest neighbors with inverted file index)

## Installation

### Prerequisites

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn
- PyYAML

### Installing FAISS

You can install either the CPU or GPU version of FAISS:

```bash
# CPU version
pip install faiss-cpu

# OR GPU version (requires CUDA)
pip install faiss-gpu
```

Note: You cannot have both versions installed simultaneously in the same environment.

### Other dependencies

```bash
pip install numpy matplotlib scikit-learn pyyaml pandas
```

## Project Structure

```
faiss_project/
│
├── main.py                 # Main entry point
├── config.yml              # Configuration file
│
├── dataset/
│   ├── __init__.py
│   ├── synthetic_data.py   # For creating dummy data
│   └── parquet_loader.py   # For loading embeddings from parquet files
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py    # For visualizing vectors and results
│   └── evaluation.py       # For evaluating search performance
│
├── models/
│   ├── __init__.py
│   ├── cosine_search.py    # Cosine similarity search implementation
│   ├── faiss_flat.py       # FAISS flat index implementation
│   └── faiss_ivf.py        # FAISS IVF index implementation
│
├── trained_models/         # Directory to save trained indexes
│   └── .gitkeep
│
└── results/                # Directory to save results and figures
    └── .gitkeep
```

## Usage

Run the main script with the default configuration:

```bash
python main.py
```

Or specify a custom configuration file:

```bash
python main.py --config my_config.yml
```

## Configuration

The `config.yml` file contains all settings for the experiments:

```yaml
# General settings
experiment:
  name: faiss_comparison
  seed: 42
  save_results: true
  visualize: true

# Data settings
data:
  use_synthetic: true
  n_vectors: 1000
  n_dimensions: 3
  parquet_path: null  # Set to a path to use parquet files instead of synthetic data

# Search settings
search:
  k: 5  # Number of nearest neighbors to retrieve
  n_queries: 1000  # Number of queries for performance testing

# FAISS IVF settings
faiss_ivf:
  n_regions: 10  # Number of clusters/regions
  nprobe: 3      # Number of regions to search
  
# Performance test settings
performance_test:
  enabled: true
  vector_sizes: [100, 1000, 10000, 100000]
  dimension: 128
```

## Key Features

1. **Modular Design**: Separate modules for data handling, search algorithms, visualization, and evaluation
2. **Multiple Search Methods**: Compare different search algorithms
3. **Performance Testing**: Evaluate how each method scales with vector count
4. **Visualization**: 3D visualization of vector spaces and search results
5. **Parquet Support**: Load real embeddings from parquet files
6. **Index Saving/Loading**: Save trained indexes for future use

## Adding Your Own Data

To use your own embedding data:

1. Set `use_synthetic: false` in the config
2. Set `parquet_path` to the path of your parquet file(s)
3. The parquet file should have a column containing embedding vectors (as lists or arrays)

## Extending the Project

To add a new search method:

1. Create a new file in the `models/` directory
2. Implement your search function
3. Update `__init__.py` to export your function
4. Add your method to the comparison in `main.py`

## License

MIT