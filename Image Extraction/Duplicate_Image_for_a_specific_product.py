# import numpy as np
# from scipy.spatial.distance import cosine
# import os

# vecs = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_vecs1.npy")
# names = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_names1.npy")

# # Flatten the vectors to remove singleton dimensions
# vecs = vecs.squeeze()

# # The threshold for similarity
# threshold = 0.9

# target_img_name = r"C:/Users/eDominer/Python Project/Products/DE2869F2DD7740B9893A7F905D65C788.jpg"  

# # Ensure the target image exists in the dataset
# if target_img_name in names:
#     target_idx = np.argwhere(names == target_img_name).flatten()[0] 
#     target_vec = vecs[target_idx] 

#     # Calculate cosine similarity
#     similarities = np.dot(vecs, target_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(target_vec))
    
#     duplicates = []
#     for i, similarity in enumerate(similarities):
#         if similarity >= threshold and i != target_idx:
#             duplicates.append(names[i])

#     if duplicates:
#         print(f"Duplicate images for {target_img_name}:")
#         for duplicate in duplicates:
#             print(duplicate)
#     else:
#         print(f"No duplicates found for {target_img_name}.")
# else:
#     print(f"Image {target_img_name} not found in the dataset.")


# import streamlit as st
# import numpy as np
# from scipy.spatial.distance import cdist
# from PIL import Image
# import os

# @st.cache_data
# def read_data():
#     all_vecs = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_vecs1.npy")
#     all_names = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_names1.npy")
#     return all_vecs, all_names

# vecs, names = read_data()

# # Flatten the vectors to remove singleton dimensions
# vecs = vecs.squeeze()

# # Define the threshold for similarity (cosine distance threshold)
# threshold = 0.1

# image_dir = r"C:/Users/eDominer/Python Project/Products/"

# target_img_name = st.selectbox("Select an image to find duplicates:", names)

# if target_img_name:
#     target_idx = np.argwhere(names == target_img_name).flatten()[0]
#     target_vec = vecs[target_idx]

#     distances = cdist(target_vec[None, :], vecs, metric='cosine').squeeze()  # Cosine distance

#     duplicates = []
#     for i, dist in enumerate(distances):
#         if dist < threshold and i != target_idx:
#             duplicates.append(names[i])

#     st.image(Image.open(os.path.join(image_dir, target_img_name)), caption=f"Selected Image: {target_img_name}")

#     if duplicates:
#         st.write(f"Found {len(duplicates)} duplicate(s) for {target_img_name}:")

#         rows = [duplicates[i:i + 2] for i in range(0, len(duplicates), 2)]
#         for row in rows:
#             cols = st.columns(len(row))
#             for idx, duplicate in enumerate(row):
#                 img_path = os.path.join(image_dir, duplicate)
#                 cols[idx].image(Image.open(img_path), caption=f"Duplicate: {duplicate}")

#     else:
#         st.write(f"No duplicates found for {target_img_name}.")

# import streamlit as st
# import numpy as np
# from PIL import Image
# import os

# @st.cache_data
# def read_data():
#     all_vecs = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_vecs1.npy")
#     all_names = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_names1.npy")
#     return all_vecs, all_names

# vecs, names = read_data()

# # Flatten the vectors to remove singleton dimensions
# vecs = vecs.squeeze()

# image_dir = r"C:/Users/eDominer/Python Project/Products/"

# target_img_name = st.selectbox("Select an image to find duplicates:", names)

# if target_img_name:
#     target_idx = np.argwhere(names == target_img_name).flatten()[0]
#     target_vec = vecs[target_idx]

#     duplicates = []
#     for i, vec in enumerate(vecs):
#         if np.array_equal(vec, target_vec) and i != target_idx:
#             duplicates.append(names[i])

#     st.image(Image.open(os.path.join(image_dir, target_img_name)), caption=f"Selected Image: {target_img_name}")

#     if duplicates:
#         st.write(f"Found {len(duplicates)} exact duplicate(s) for {target_img_name}:")

#         rows = [duplicates[i:i + 2] for i in range(0, len(duplicates), 2)]
#         for row in rows:
#             cols = st.columns(len(row))
#             for idx, duplicate in enumerate(row):
#                 img_path = os.path.join(image_dir, duplicate)
#                 cols[idx].image(Image.open(img_path), caption=f"Duplicate: {duplicate}")

#     else:
#         st.write(f"No exact duplicates found for {target_img_name}.")



import streamlit as st
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
import os

@st.cache_data
def read_data():
    all_vecs = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_vecs1.npy")
    all_names = np.load(f"C:/Users/eDominer/Python Project/Image Extraction/all_names1.npy")
    return all_vecs, all_names

vecs, names = read_data()

# Flatten the vectors to remove singleton dimensions
vecs = vecs.squeeze()

# Define the similarity threshold
similarity_threshold = 0.96
percentage = similarity_threshold*100;

image_dir = r"C:/Users/eDominer/Python Project/Products/"

target_img_name = st.selectbox("Select an image to find duplicates:", names)

if target_img_name:
    target_idx = np.argwhere(names == target_img_name).flatten()[0]
    target_vec = vecs[target_idx]

    # Calculate cosine similarity
    similarities = np.dot(vecs, target_vec) / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(target_vec))

    duplicates = []
    for i, similarity in enumerate(similarities):
        if similarity >= similarity_threshold and i != target_idx:
            duplicates.append(names[i])

    st.image(Image.open(os.path.join(image_dir, target_img_name)), caption=f"Selected Image: {target_img_name}")

    if duplicates:
        st.write(f"Found {len(duplicates)} similar item(s) (≥{percentage}%) for {target_img_name}:")

        rows = [duplicates[i:i + 2] for i in range(0, len(duplicates), 2)]
        for row in rows:
            cols = st.columns(len(row))
            for idx, duplicate in enumerate(row):
                img_path = os.path.join(image_dir, duplicate)
                cols[idx].image(Image.open(img_path), caption=f"Duplicate: {duplicate}")

    else:
        st.write(f"No similar items (≥{percentage}%) found for {target_img_name}.")


