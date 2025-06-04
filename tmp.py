import numpy as np
import faiss

# Example with d=90, m=3 sub-quantizers, 8 bits (256 centroids per sub-quantizer)
d = 90
m = 3  # Number of sub-quantizers
bits = 8  # 2^8 = 256 centroids per sub-quantizer

# Create sample data
n_vectors = 1000
vectors = np.random.random((n_vectors, d)).astype('float32')

# Create PQ index
index = faiss.IndexPQ(d, m, bits)

# Train the index (this creates the sub-codebooks)
index.train(vectors)

# Add vectors (this encodes them into PQ codes)
index.add(vectors)

# Let's examine what happens to a single vector
test_vector = vectors[0]
print(f"Original vector shape: {test_vector.shape}")
print(f"Original vector (first 10 dims): {test_vector[:10]}")

# Get the PQ codes for this vector
codes = np.zeros((1, m), dtype=np.uint8)
index.sa_encode(test_vector.reshape(1, -1), codes)
print(f"PQ codes: {codes[0]}")  # This gives you [c1, c10, c200] equivalent

# Reconstruct the vector from codes
reconstructed = np.zeros((1, d), dtype='float32')
index.sa_decode(codes, reconstructed)
print(f"Reconstructed vector (first 10 dims): {reconstructed[0][:10]}")

# Calculate reconstruction error
error = np.linalg.norm(test_vector - reconstructed[0])
print(f"Reconstruction error: {error:.4f}")