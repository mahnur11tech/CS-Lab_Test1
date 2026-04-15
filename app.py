import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
from gtts import gTTS
import os
import tempfile

# Page Configuration
st.set_page_config(page_title="AI Vision Pro", layout="wide")
st.title("🤖 AI Local Vision & Speech App")

# Load Pre-trained YOLOv8 Model (Free & Local)
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Nano version for speed

model = load_model()

# Sidebar for Navigation
option = st.sidebar.selectbox("Select Mode", ("File Upload", "URL Image", "Live Detection"))

def speak_text(text):
    """Convert text to speech and play in streamlit"""
    if text:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

def process_image(img):
    """Run YOLO detection and return results"""
    results = model(img)
    # Get detected object names
    names = [model.names[int(c)] for c in results[0].boxes.cls]
    detected_str = ", ".join(set(names)) if names else "No objects detected"
    
    # Plot results on image
    res_plotted = results[0].plot()
    return res_plotted, detected_str

# --- UI LOGIC ---

if option == "File Upload":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze & Speak"):
            res_img, text = process_image(image)
            st.image(res_img, caption="Processed Image", use_container_width=True)
            st.success(f"Detected: {text}")
            speak_text(f"I can see {text}")

elif option == "URL Image":
    url = st.text_input("Enter Image URL:")
    if url:
        try:
            st.image(url, caption="URL Image", use_container_width=True)
            if st.button("Analyze & Speak"):
                res_img, text = process_image(url)
                st.image(res_img, caption="Processed Image", use_container_width=True)
                st.success(f"Detected: {text}")
                speak_text(f"I can see {text}")
        except Exception as e:
            st.error("Invalid URL or Image format.")

elif option == "Live Detection":
    st.warning("Webcam live stream requires 'streamlit-webrtc' for cloud deployment, but this local version uses OpenCV.")
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    cam = cv2.VideoCapture(0)

    while run:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)
        res_plotted = results[0].plot()
        FRAME_WINDOW.image(res_plotted)
    else:
        cam.release()
        st.write("Webcam Stopped.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses YOLOv8 & gTTS (Local Models). No API keys needed!")
