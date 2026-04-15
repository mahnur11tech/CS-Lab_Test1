import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os

# Page Config
st.set_page_config(page_title="AI Vision", layout="centered")
st.title("🤖 All-in-One AI Vision App")

# Load Model (Caches so it doesn't download every time)
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

model = load_yolo()

def speak(text):
    if text and text != "No objects detected":
        tts = gTTS(text=f"I can see {text}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

# Tabs for cleaner UI
tab1, tab2 = st.tabs(["📤 Upload & URL", "🎥 Live Camera"])

with tab1:
    source = st.radio("Select Source:", ("Local Upload", "Image URL"))
    img_input = None
    
    if source == "Local Upload":
        img_input = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    else:
        url = st.text_input("Paste Image URL here:")
        if url: img_input = url

    if img_input:
        if st.button("Analyze Now"):
            # Process
            results = model(img_input)
            res_img = results[0].plot()
            
            # Detect names
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "No objects detected"
            
            # Show Results
            st.image(res_img, caption=f"Detected: {labels}", use_container_width=True)
            st.success(f"Found: {labels}")
            speak(labels)

with tab2:
    st.info("Note: Live detection works best on Local PC. On Cloud, use 'Camera Input' below.")
    cam_image = st.camera_input("Take a photo to analyze")
    
    if cam_image:
        img = Image.open(cam_image)
        results = model(img)
        res_img = results[0].plot()
        
        names = [model.names[int(c)] for c in results[0].boxes.cls]
        labels = ", ".join(set(names)) if names else "Nothing detected"
        
        st.image(res_img)
        speak(labels)

st.divider()
st.caption("Running on YOLOv8 & gTTS | No API Required")
