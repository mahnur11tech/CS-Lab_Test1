import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Object Detector", layout="centered")

st.title("🤖 Light-Weight AI Detector")
st.write("Duniya ki sabse halki AI app jo memory full nahi karegi.")

# --- INITIALIZE MEDIAPIPE ---
mp_object_detection = mp.solutions.object_detection
mp_drawing = mp.solutions.drawing_utils

# --- UI LOGIC ---
source = st.radio("Image kahan se layen?", ["Upload Karein", "Camera Se Lein"])

img_file = None
if source == "Upload Karein":
    img_file = st.file_uploader("Image select karein", type=['jpg', 'png', 'jpeg'])
else:
    img_file = st.camera_input("Photo khinchen")

if img_file:
    # Convert image for processing
    image = Image.open(img_file)
    image_np = np.array(image)
    
    st.image(image, caption="Aapki Image", use_container_width=True)

    if st.button("AI Se Check Karwayen"):
        with st.spinner("AI dekh raha hai..."):
            # Setup MediaPipe Detector
            with mp_object_detection.ObjectDetection(min_detection_confidence=0.5) as detector:
                results = detector.process(image_np)

                # Draw Results
                annotated_image = image_np.copy()
                detections_count = 0
                
                if results.detections:
                    for detection in results.detections:
                        mp_drawing.draw_detection(annotated_image, detection)
                        detections_count += 1
                
                # Show Result
                st.image(annotated_image, caption="AI Result", use_container_width=True)
                
                if detections_count > 0:
                    st.success(f"AI ne {detections_count} cheezain pehchani hain!")
                else:
                    st.warning("AI ko kuch nazar nahi aaya.")

st.info("Note: Is app mein sirf 2 files hain aur ye memory crash nahi hogi.")
