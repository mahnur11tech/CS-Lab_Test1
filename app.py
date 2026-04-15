import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile

# --- APP CONFIG ---
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Smart AI Vision & Voice Assistant")

# --- MODEL LOADING ---
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
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except Exception as e:
            st.warning("Voice output failed, but detection is complete.")

# --- UI ---
st.sidebar.header("Configuration")
source = st.sidebar.radio("Select Image Source:", ("Local Upload", "Image URL", "Live Camera"))

img_input = None
if source == "Local Upload":
    img_input = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
elif source == "Image URL":
    url = st.text_input("Paste Image URL:")
    if url: img_input = url
elif source == "Live Camera":
    img_input = st.camera_input("Take a snapshot")

if img_input:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_input, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner('AI is thinking...'):
            results = model(img_input)
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) 
            
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("AI Result")
                st.image(res_image, use_container_width=True)
                st.success(f"**Detected:** {labels}")
                speak(labels)
