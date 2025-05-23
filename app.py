import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import os

MODEL_PATH = "digit_model.h5"

# Load model and scaler
if os.path.exists(MODEL_PATH):
    model, scaler = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found. Please train and save it as 'digit_model.h5'.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognition ")
st.markdown("Draw a digit (0–9) below in the box and click **Predict**.")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Ink color
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

if canvas_result.image_data is not None:
    # Preprocess the image
    img = canvas_result.image_data.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized
    normalized = inverted / 255.0
    flat = normalized.flatten().reshape(1, -1)

    # Scale using the same scaler used during training
    flat_scaled = scaler.transform(flat)

    st.image(resized, width=150, caption="🧪 Processed 28x28 Image")

    if st.button("📌 Predict"):
        prediction = model.predict(flat_scaled)
        st.success(f"✅ Predicted Digit: **{prediction[0]}**")

st.caption(" Draw using your mouse or touchscreen.")
