import numpy as np
from scipy.spatial.distance import cdist
import os

# Load the precomputed feature vectors and names
vecs = np.load("all_vecs.npy")
names = np.load("all_names.npy")

# Flatten the vectors to remove singleton dimensions
vecs = vecs.squeeze()  # Now it will be (N, D)
threshold = 0.1  # You can adjust this threshold based on your needs

# Calculate pairwise distances (Cosine similarity or Euclidean)
# Here we use cosine similarity (1 - cosine similarity = distance)
distances = cdist(vecs, vecs, metric='cosine')  # Cosine distance will be between 0 and 2
# Alternatively, use Euclidean distance: distances = cdist(vecs, vecs, metric='euclidean')

# Identify duplicates based on the distance threshold
duplicates = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):  # Only compare each pair once
        if distances[i, j] < threshold:  # If distance is below threshold, they are duplicates
            duplicates.append((names[i], names[j]))

# Print or save the duplicate image pairs
if duplicates:
    print("Duplicate images found:")
    for img1, img2 in duplicates:
        print(f"Duplicate Pair: {img1} - {img2}")
else:
    print("No duplicates found.")
