import streamlit as st
import numpy as np
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import joblib
import os
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelBinarizer

MODEL_PATH = "sklearn_digit_model.pkl"

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    # Basic fallback model (if no pre-trained model found)
    model = SGDClassifier()
    st.warning("No pre-trained model found. An empty SGDClassifier has been initialized.")

# Streamlit UI
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("✍️ Handwritten Digit Recognition (No TensorFlow)")
st.markdown("Draw a digit (0–9) below and click **Predict**.")

canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=300,
    width=300,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Preprocess image
    img = canvas_result.image_data.astype("uint8")
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    inverted = 255 - resized
    scaled = inverted / 16.0
    flat = scaled.flatten().reshape(1, -1)

    # Show image
    st.image(resized, width=150, caption="🧪 Processed 8x8 Image")

    if st.button("📌 Predict"):
        try:
            prediction = model.predict(flat)
            st.success(f"✅ Predicted Digit: **{prediction[0]}**")

            # Feedback section
            is_correct = st.radio("Is the prediction correct?", ("Yes", "No"))
            if is_correct == "No":
                correct_label = st.number_input("Enter the correct digit:", 0, 9, step=1)
                if st.button("🔄 Retrain with Correct Label"):
                    model.partial_fit(flat, [correct_label], classes=np.arange(10))
                    joblib.dump(model, MODEL_PATH)
                    st.success("Model updated with new feedback!")
                    new_pred = model.predict(flat)
                    st.info(f"🆕 New Prediction: {new_pred[0]}")
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

st.caption("🎨 Use mouse or touchscreen to draw.")
