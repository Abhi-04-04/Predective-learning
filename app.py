import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Load or build model
model_path = "mnist_cnn_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    # Simple CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Streamlit UI
st.title("🖊️ Handwritten Digit Recognition")
st.subheader("Draw a digit (0–9) below:")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    image = canvas_result.image_data

    # Preprocess image
    img = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_RGBA2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)

    if st.button("Predict", key="predict_btn"):
        pred = model.predict(img)
        predicted_label = np.argmax(pred)
        st.success(f"Predicted Label: {predicted_label}")

        is_correct = st.radio("Is the prediction correct?", ('Yes', 'No'), key="feedback_radio")

        if is_correct == 'No':
            correct_label = st.number_input("Enter the correct digit:", min_value=0, max_value=9, step=1, key="correct_input")
            if st.button("Train with Correct Label", key="train_btn"):
                y_correct = to_categorical([correct_label], num_classes=10)
                model.fit(img, y_correct, epochs=2, verbose=0)
                model.save(model_path)
                st.success(f"Model retrained with correct label {correct_label}.")
                new_pred = np.argmax(model.predict(img))
                st.info(f"New Prediction: {new_pred}")

    # Show preprocessed image
    st.subheader("Preprocessed Image:")
    st.image(img.reshape(28, 28), width=150, clamp=True)







