import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import gdown
import os

# 1. Download model from Google Drive if not present
model_path = "pneumonia_cnn_model.h5"
gdrive_file_id = "1nP1sWihrPi8v9pAdFMpQDBkUTycm_17t"  # 🔁 Replace with your actual Google Drive file ID

if not os.path.exists(model_path):
    with st.spinner("Downloading model... Please wait."):
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, model_path, quiet=False)

# 2. Load model
model = load_model(model_path)

# 3. Streamlit config
st.set_page_config(page_title="Pneumonia Detection", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🩺 Pneumonia Detection from Chest X-ray</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a chest X-ray image to predict whether it shows signs of PNEUMONIA or is NORMAL.</p>", unsafe_allow_html=True)

# 4. File uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    img = Image.open(uploaded_file)

    # Preprocess the image
    img_resized = img.convert("RGB").resize((150, 150))
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
    confidence = prediction if label == "PNEUMONIA" else 1 - prediction
    confidence_percent = confidence * 100
    color = 'red' if label == 'PNEUMONIA' else 'green'

    # Display result using matplotlib
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(f"{label} ({confidence_percent:.2f}% confidence)", fontsize=14, color=color)
    st.pyplot(fig)

# Optional footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Developed by Manmeet Singh | CDAC Mohali</p>", unsafe_allow_html=True)
