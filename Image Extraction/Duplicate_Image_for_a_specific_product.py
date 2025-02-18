# import numpy as np
# from scipy.spatial.distance import cdist
# import os

# # Load the precomputed feature vectors and names
# vecs = np.load("all_vecs.npy")
# names = np.load("all_names.npy")

# # Flatten the vectors to remove singleton dimensions
# vecs = vecs.squeeze()  # Now it will be (N, D)

# # Define the threshold for similarity
# threshold = 0.1  # You can adjust this threshold based on your needs

# # Specify the target image for which you want to find duplicates
# target_img_name = r"C:/Users/eDominer/Python Project/Products/T_FB097A0D761B4C27A3FEB616ADDE3161.jpg"  # Replace with your image name

# # Ensure the target image exists in the dataset
# if target_img_name in names:
#     target_idx = np.argwhere(names == target_img_name).flatten()[0]  # Get index of the target image
#     target_vec = vecs[target_idx]  # Get the feature vector for the target image

#     # Calculate pairwise distances between the target image and all other images
#     distances = cdist(target_vec[None, :], vecs, metric='cosine').squeeze()  # Cosine distance
    
#     # Find images with distances below the threshold (duplicates)
#     duplicates = []
#     for i, dist in enumerate(distances):
#         if dist < threshold and i != target_idx:  # Don't consider the target image itself
#             duplicates.append(names[i])

#     # Print or return the duplicate image names
#     if duplicates:
#         print(f"Duplicate images for {target_img_name}:")
#         for duplicate in duplicates:
#             print(duplicate)
#     else:
#         print(f"No duplicates found for {target_img_name}.")
# else:
#     print(f"Image {target_img_name} not found in the dataset.")


import streamlit as st
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image
import os

# Load the precomputed feature vectors and names
@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs, all_names

# Load the data
vecs, names = read_data()

# Flatten the vectors to remove singleton dimensions
vecs = vecs.squeeze()  # Now it will be (N, D)

# Define the threshold for similarity (cosine distance threshold)
threshold = 0.1  # Adjust as needed

# File path for images
image_dir = r"C:/Users/eDominer/Python Project/Products/"

# Allow the user to select a target image
target_img_name = st.selectbox("Select an image to find duplicates:", names)

# Ensure the target image exists in the dataset
if target_img_name:
    # Find the index of the selected image in the dataset
    target_idx = np.argwhere(names == target_img_name).flatten()[0]
    target_vec = vecs[target_idx]  # Get the feature vector for the selected image

    # Calculate pairwise distances between the selected image and all other images
    distances = cdist(target_vec[None, :], vecs, metric='cosine').squeeze()  # Cosine distance

    # Find images with distances below the threshold (duplicates)
    duplicates = []
    for i, dist in enumerate(distances):
        if dist < threshold and i != target_idx:  # Exclude the target image itself
            duplicates.append(names[i])

    # Display the selected image
    st.image(Image.open(os.path.join(image_dir, target_img_name)), caption=f"Selected Image: {target_img_name}")

    if duplicates:
        # Display the duplicate images
        st.write(f"Found {len(duplicates)} duplicate(s) for {target_img_name}:")
        cols = st.columns(min(len(duplicates), 5))  # Max 5 images per row
        
        for i, duplicate in enumerate(duplicates):
            img_path = os.path.join(image_dir, duplicate)
            cols[i % 5].image(Image.open(img_path), caption=f"Duplicate: {duplicate}")
    else:
        st.write(f"No duplicates found for {target_img_name}.")
