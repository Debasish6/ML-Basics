import cv2
import tensorflow as tf
import numpy as np

def resize_image(image, target_size=(256, 256)):
    return cv2.resize(image, target_size)

def load_and_preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image)
    image = image / 255.0
    return image

def load_and_preprocess_from_path_label(path, xmin, ymin, xmax, ymax, label):
    return load_and_preprocess_image(path), (xmin, ymin, xmax, ymax, label)

# Replace these paths with your actual image paths
paths = [r"c:/Users/edominer/Python Project/Extracting Text from Invoice/Edominer-1201_page-1.jpg", r"c:/Users/edominer/Python Project/Extracting Text from Invoice/invoice.png"]
xmins = [100, 200]
ymins = [100, 200]
xmaxs = [150, 250]
ymaxs = [150, 250]
labels = [1, 0]

dataset = tf.data.Dataset.from_tensor_slices((paths, xmins, ymins, xmaxs, ymaxs, labels))
dataset = dataset.map(lambda path, xmin, ymin, xmax, ymax, label: 
                      tf.py_function(func=load_and_preprocess_from_path_label, inp=[path, xmin, ymin, xmax, ymax, label], 
                                    Tout=[tf.float32, tf.float32]))
dataset = dataset.shuffle(buffer_size=1000).batch(4).prefetch(tf.data.experimental.AUTOTUNE)

from tensorflow.keras import layers, models

input_shape = (256, 256, 3)

model = models.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4)
])

tf.keras.mixed_precision.set_global_policy('mixed_float16')

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.fit(dataset, epochs=10)
print("Model training complete.")
