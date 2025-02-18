import streamlit as st
import numpy as np
from PIL import Image
import os
from scipy.spatial.distance import cdist

# Function to read the precomputed vectors and names
@st.cache_data
def read_data():
    all_vecs = np.load("all_vecs.npy")
    all_names = np.load("all_names.npy")
    return all_vecs, all_names

vecs, names = read_data()

# Check the shape of the loaded vectors
st.write("Shape of vecs:", vecs.shape)  # This should print (N, D)

# Define the directory where images are stored
image_dir = r"C:/Users/eDominer/Python Project/Products/"

# Create columns for layout
_, fcol2, _ = st.columns(3)
scol1, scol2 = st.columns(2)

# Buttons for starting and finding similar images
ch = scol1.button("Start / Change")
fs = scol2.button("Find Similar")

# Display a random image when "Start / Change" is clicked
if ch:
    random_name = names[np.random.randint(len(names))]
    image_path = os.path.join(image_dir, random_name)
    fcol2.image(Image.open(image_path))
    st.session_state["disp_img"] = random_name
    st.write(st.session_state["disp_img"])

# Display the top 5 similar images when "Find Similar" is clicked
if fs:
    c1, c2, c3, c4, c5 = st.columns(5)

    # Ensure the "disp_img" state exists before using it
    if "disp_img" in st.session_state:
        target_img_name = st.session_state["disp_img"]
        
        # Fixing np.argwhere to get a single index
        idx = np.argwhere(names == target_img_name).flatten()[0]  # Extracting the scalar index
        
        target_vec = vecs[idx]  # Get the corresponding feature vector for the target image

        # Debugging step: Check the shape of target_vec
        st.write(f"Shape of target_vec for {target_img_name}: {target_vec.shape}")  # This should print (512, 1, 1)

        # Ensure target_vec is a 1D array (512,)
        target_vec = target_vec.squeeze()  # Remove singleton dimensions

        # Debugging step: Check the shape after squeezing
        st.write(f"Shape of target_vec after squeezing: {target_vec.shape}")  # This should print (512,)

        # Ensure vecs is also squeezed
        vecs = vecs.squeeze()  # Remove singleton dimensions

        # Debugging step: Check the shape of vecs after squeezing
        st.write(f"Shape of vecs after squeezing: {vecs.shape}")  # This should print (N, 512)

        # Load and display the target image
        target_image_path = os.path.join(image_dir, target_img_name)
        fcol2.image(Image.open(target_image_path))

        # Find top 5 most similar images using cdist
        top5 = cdist(target_vec[None, :], vecs).squeeze().argsort()[1:6]  # target_vec[None, :] reshapes it to (1, 512)

        # Debugging step: Check the top 5 indices
        st.write(f"Top 5 indices: {top5}")

        # Display the top 5 similar images
        for i, col in enumerate([c1, c2, c3, c4, c5]):
            similar_img_name = names[top5[i]]
            similar_img_path = os.path.join(image_dir, similar_img_name)
            col.image(Image.open(similar_img_path))
