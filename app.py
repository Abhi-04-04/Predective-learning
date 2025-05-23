import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import os

MODEL_PATH = "sklearn_digit_model.pkl"

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("❌ Model file not found. Please train and save it as 'sklearn_digit_model.pkl'.")
    st.stop()

# Streamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognition (No TensorFlow)")
st.markdown("Draw a digit (0–9) below in the box and click **Predict**.")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",  # Ink color
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized
    scaled = inverted / 16.0
    flat = scaled.flatten().reshape(1, -1)

    st.image(resized, width=150, caption="🧪 Processed 8x8 Image")

    if st.button("📌 Predict"):
        prediction = model.predict(flat)
        st.success(f"✅ Predicted Digit: **{prediction[0]}**")

st.caption("🎨 Draw using your mouse or touchscreen.")
