import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile
import numpy as np

# Page Setup
st.set_page_config(page_title="AI Vision", layout="centered")
st.title("🤖 Local AI Vision (No API)")

# Model Loading (Cached for speed)
@st.cache_resource
def load_model():
    # Download tiny model for faster performance on Cloud
    return YOLO('yolov8n.pt')

model = load_model()

def speak(text):
    """Text to Speech logic"""
    if text:
        tts = gTTS(text=f"Detected {text}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

# UI Tabs
tab1, tab2 = st.tabs(["🖼️ Image Upload/URL", "📸 Camera Input"])

with tab1:
    choice = st.radio("Source:", ["Upload File", "Image URL"])
    img_data = None
    
    if choice == "Upload File":
        img_data = st.file_uploader("Select Image", type=['jpg', 'jpeg', 'png'])
    else:
        url = st.text_input("Enter Image URL")
        if url: img_data = url

    if img_data:
        if st.button("Analyze Image"):
            results = model(img_data)
            # Plotting with YOLO's internal PIL-based method
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Result", use_container_width=True)
            
            # Extract labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            label_text = ", ".join(set(names)) if names else "Nothing"
            st.success(f"Labels: {label_text}")
            speak(label_text)

with tab2:
    st.write("Take a picture using your browser camera:")
    cam_photo = st.camera_input("Snapshot")
    
    if cam_photo:
        # Process the camera photo
        img = Image.open(cam_photo)
        results = model(img)
        res_plotted = results[0].plot()
        st.image(res_plotted)
        
        names = [model.names[int(c)] for c in results[0].boxes.cls]
        label_text = ", ".join(set(names)) if names else "Nothing"
        speak(label_text)

st.divider()
st.info("Direct Run: 2 Files Only (app.py & requirements.txt)")
