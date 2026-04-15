import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Vision Assistant", layout="wide")
st.title("🤖 AI Vision & Voice Assistant")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the smallest model (~6MB)
    return YOLO('yolov8n.pt')

model = load_model()

def speak(text):
    if text and text != "Nothing detected":
        try:
            tts = gTTS(text=f"I see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            st.error("Audio error.")

# --- UI ---
source = st.sidebar.radio("Select Source:", ("Upload", "URL", "Camera"))
img_input = None

if source == "Upload":
    img_input = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
elif source == "URL":
    url = st.text_input("Paste Image URL:")
    if url: img_input = url
elif source == "Camera":
    img_input = st.camera_input("Take a photo")

if img_input:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(img_input, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner('Analyzing...'):
            results = model(img_input)
            
            # Plot Results
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR to RGB
            
            # Labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("AI Result")
                st.image(res_image, use_container_width=True)
                st.success(f"Detected: {labels}")
                speak(labels)
