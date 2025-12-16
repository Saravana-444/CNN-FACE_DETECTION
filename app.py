import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# ---------------- CONFIG ----------------
MODEL_URL = "https://drive.google.com/uc?id=1BQAWXyu7FcMq7rbKG6D5CJBuUCis7ukk"
MODEL_PATH = "face_recognition_mobilenet_3class.h5"
IMG_SIZE = 224

CLASS_NAMES = ["Gobinath", "Guru Nagajothi", "Saravana kumar"]
# MUST MATCH train_data.class_indices ORDER

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ---------------- UI ----------------
st.title("🧠 Face Recognition App")
st.write("Upload a face image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"### Predicted Person: {CLASS_NAMES[class_index]}")
    st.info(f"Confidence: {confidence:.2f}%")

    st.write("Raw prediction:", prediction)
