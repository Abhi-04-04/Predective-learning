# utils.py

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

def preprocess_image(image):
    # Resize to 28x28, convert to grayscale
    image = image.resize((28, 28)).convert("L")
    # Invert image (MNIST has white digits on black)
    image = ImageOps.invert(image)
    # Normalize and reshape
    image_array = np.array(image).astype("float32") / 255.0
    return image_array.reshape(1, 28, 28, 1)

def load_model(path="mnist_cnn_model.h5"):
    return tf.keras.models.load_model(path)
