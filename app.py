import streamlit as st
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from PIL import Image
import numpy as np
from gtts import gTTS
import tempfile
import os

# Page Setup
st.set_page_config(page_title="Light AI Vision", layout="centered")
st.title("🚀 Ultra-Light AI Vision")
st.write("Using a memory-efficient model to prevent 'Out of Memory' errors.")

def speak(text):
    if text:
        tts = gTTS(text=f"I can see {text}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

# Choice
option = st.radio("Select Source:", ["Upload Image", "Camera Input"])

img_input = None
if option == "Upload Image":
    img_input = st.file_uploader("Upload", type=['jpg', 'png', 'jpeg'])
else:
    img_input = st.camera_input("Take a photo")

if img_input:
    # Read Image
    file_bytes = np.asarray(bytearray(img_input.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    st.image(image_rgb, caption="Input Image", use_container_width=True)

    if st.button("Identify Objects"):
        with st.spinner("Processing..."):
            # Object Detection (Super Light)
            bbox, label, conf = cv.detect_common_objects(image)
            
            # Draw boxes
            out_img = draw_bbox(image, bbox, label, conf)
            out_img_rgb = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            
            # Results
            st.image(out_img_rgb, caption="AI Result", use_container_width=True)
            
            unique_labels = ", ".join(set(label)) if label else "Nothing detected"
            st.success(f"Detected: {unique_labels}")
            
            # Audio
            speak(unique_labels)

st.divider()
st.caption("Running on CVLib & OpenCV (No Heavy Torch/YOLO)")
