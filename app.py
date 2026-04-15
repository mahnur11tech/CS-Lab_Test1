import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile
import io

# Page Config
st.set_page_config(page_title="AI Vision", layout="wide")
st.title("🚀 Pure AI Vision App")

# Model Load (Directly from Ultralytics)
@st.cache_resource
def load_model():
    # yolov8n.pt is the lightest model
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model loading error. This usually happens on Cloud without 'packages.txt'.")

def speak(text):
    if text:
        tts = gTTS(text=f"I found {text}", lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            st.audio(fp.name, format="audio/mp3")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    source = st.radio("Choose Source:", ["Camera", "Upload", "URL"])
    
    img_input = None
    if source == "Camera":
        img_input = st.camera_input("Take a photo")
    elif source == "Upload":
        img_input = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    else:
        url = st.text_input("Image URL")
        if url: img_input = url

with col2:
    st.header("Result")
    if img_input:
        # Prediction
        results = model(img_input)
        
        # Plotting using PIL instead of OpenCV
        res_array = results[0].plot() # This returns a numpy array (BGR)
        res_image = Image.fromarray(res_array[..., ::-1]) # Convert BGR to RGB
        
        st.image(res_image, use_container_width=True)
        
        # Get Names
        names = [model.names[int(c)] for c in results[0].boxes.cls]
        labels = ", ".join(set(names)) if names else "Nothing detected"
        
        st.success(f"Detected: {labels}")
        if st.button("Listen to Result"):
            speak(labels)

st.divider()
st.caption("Standard 2-file deployment: app.py & requirements.txt")
