import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Vision Pro", layout="wide")
st.title("🤖 AI Vision & Voice Assistant")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the smallest model (only ~6MB)
    return YOLO('yolov8n.pt')

model = load_model()

def speak(text):
    if text and text != "Nothing detected":
        try:
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            pass

# --- UI NAVIGATION ---
source = st.sidebar.radio("Select Source:", ("Upload", "URL", "Camera"))
img_input = None

if source == "Upload":
    img_input = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
elif source == "URL":
    url = st.text_input("Paste Image URL:")
    if url: img_input = url
elif source == "Camera":
    img_input = st.camera_input("Take a photo")

# --- EXECUTION ---
if img_input:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(img_input, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner('AI is analyzing...'):
            results = model(img_input)
            
            # Plot Results
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR to RGB
            
            # Get Labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("AI Result")
                st.image(res_image, use_container_width=True)
                st.success(f"Detected: {labels}")
                speak(labels)
