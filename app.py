import os
import sys
import subprocess

# Manual Installation setup
def install_and_import(package):
    try:
        __import__(package)
    except ImportError:
        # Agar library nahi milti to install karega
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Zaroori libraries install karen (Bina requirement file k direct install)
install_and_import('ultralytics')
install_and_import('opencv-python-headless')
install_and_import('pillow')

# --- Main App Code ---
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="AI Computer Vision", layout="centered")

st.title("🔍 AI Object Detector")
st.info("Pehli baar run hone mein 1-2 minute lag sakte hain (AI Model download ho raha hai)...")

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt") 

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_container_width=True)
    
    if st.button('Start Detection'):
        results = model.predict(source=image, conf=0.4)
        res_plotted = results[0].plot()
        st.image(res_plotted, caption='Detection Result', use_container_width=True)
        
        # Object list
        names = results[0].names
        detected = [names[int(c)] for c in results[0].boxes.cls]
        st.write(f"**Detected:** {', '.join(set(detected)) if detected else 'Nothing found'}")
