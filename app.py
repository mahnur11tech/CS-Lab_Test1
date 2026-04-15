import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from gtts import gTTS
import tempfile

# 1. OS Settings to avoid GUI errors
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 2. Page Setup
st.set_page_config(page_title="AI Vision Assistant", layout="centered")
st.title("🤖 AI Vision Assistant")

# 3. Model Loading (Cached)
@st.cache_resource
def load_model():
    # Nano model is essential for Cloud memory limits
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"AI Model Error: {e}")

# 4. Voice Function
def speak(text):
    if text and text != "Nothing":
        try:
            tts = gTTS(text=f"I found {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            pass

# 5. UI Logic
source = st.radio("Choose Source:", ["Camera", "Upload File"])

img_file = None
if source == "Camera":
    img_file = st.camera_input("Take a photo")
else:
    img_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

if img_file:
    # Open image with PIL
    img = Image.open(img_file)
    st.image(img, caption="Target Image", use_container_width=True)

    if st.button("Start AI Analysis"):
        with st.spinner("AI is thinking..."):
            # Run Detection
            results = model(img)
            
            # Process Results (Get labels)
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            labels = ", ".join(set(names)) if names else "Nothing"
            
            # Show Plotted Result
            # Note: We use YOLO's internal plot() which works with PIL
            res_array = results[0].plot() 
            res_img = Image.fromarray(res_array)
            
            st.image(res_img, caption="Detection Result", use_container_width=True)
            st.success(f"Detected: {labels}")
            speak(labels)

st.divider()
st.caption("Running on YOLOv8 & gTTS (Free Version)")
