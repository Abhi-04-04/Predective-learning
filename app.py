# app.py
import streamlit as st
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import joblib
import os

st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognition (Scikit-learn)")
st.markdown("Draw a digit (0–9) below and click **Predict**.")

# Load the model
MODEL_PATH = "sklearn_digit_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    st.error("Model file not found. Please run training script.")
    st.stop()

# Drawing canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized
    scaled = (inverted / 16.0).clip(0, 1)  # scale to match training
    flat = scaled.flatten().reshape(1, -1)

    st.image(resized, width=120, caption="Processed 8x8 Input")

    if st.button("📌 Predict"):
        prediction = model.predict(flat)[0]
        st.success(f"✅ Predicted Digit: **{prediction}**")

st.caption("🖌️ Use your mouse or touchscreen to draw. Accuracy depends on drawing clarity.")
