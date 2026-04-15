import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile
import os

os.environ["QT_QPA_PLATFORM"] = "offscreen" # Graphics error se bachne ke liye

import streamlit as st
from ultralytics import YOLO
# Baki code...
# --- APP CONFIG ---
st.set_page_config(page_title="AI Vision Pro", page_icon="🤖", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Smart AI Vision & Voice Assistant")
st.write("Upload an image, provide a URL, or use your camera for real-time detection.")

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')  # Local pre-trained model

model = load_model()

def speak(text):
    """Text to Speech function"""
    if text and text != "Nothing detected":
        tts = gTTS(text=f"I can see {text}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Configuration")
source = st.sidebar.radio("Select Image Source:", ("Local Upload", "Image URL", "Live Camera"))

img_input = None

# --- MAIN LOGIC ---
if source == "Local Upload":
    img_input = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

elif source == "Image URL":
    url = st.text_input("Paste Image URL:")
    if url:
        img_input = url

elif source == "Live Camera":
    img_input = st.camera_input("Take a snapshot")

# Analysis Section
if img_input:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img_input, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner('AI is thinking...'):
            # Run YOLO
            results = model(img_input)
            
            # Get Result Image
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1]) # BGR to RGB
            
            # Get Labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("AI Result")
                st.image(res_image, use_container_width=True)
                st.success(f"**Detected:** {labels}")
                speak(labels)

st.sidebar.markdown("---")
st.sidebar.info("App powered by YOLOv8 & Streamlit. No API Keys required!")
