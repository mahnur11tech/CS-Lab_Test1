import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from PIL import Image
import tempfile

# Force light-weight imports
try:
    from ultralytics import YOLO
    from gtts import gTTS
except ImportError:
    st.error("Libraries are still installing. Please wait 2-3 minutes and refresh.")

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Vision Assistant", layout="wide")
st.title("🤖 Smart AI Vision & Voice")

# --- MODEL LOADING (Caches the model to save memory) ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model Load Error: {e}")

def speak(text):
    if text and text != "Nothing detected":
        try:
            tts = gTTS(text=f"Detected: {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            pass

# --- UI LOGIC ---
source = st.sidebar.radio("Input Mode:", ("Upload", "URL", "Camera"))
img_input = None

if source == "Upload":
    img_input = st.file_uploader("Upload Image", type=['jpg','png','jpeg'])
elif source == "URL":
    url = st.text_input("Enter URL:")
    if url: img_input = url
elif source == "Camera":
    img_input = st.camera_input("Take Snapshot")

if img_input:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_input, caption="Original", use_container_width=True)
    
    if st.button("Analyze & Speak"):
        with st.spinner("AI Processing..."):
            results = model(img_input)
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing detected"
            
            with col2:
                st.image(res_image, caption="AI Result", use_container_width=True)
                st.success(f"Labels: {labels}")
                speak(labels)
