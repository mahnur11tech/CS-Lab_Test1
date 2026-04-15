import os
import tempfile
from PIL import Image
import streamlit as st
from gtts import gTTS
from ultralytics import YOLO

# --- OS SETTINGS (To prevent crashes on Cloud) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Vision Assistant", page_icon="🤖", layout="wide")

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    .stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #FF4B4B; color: white; font-weight: bold; }
    .stSuccess { background-color: #e8f5e9; }
    </style>
    """, unsafe_allow_html=True)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the nano model (lightest & fastest for cloud)
    return YOLO('yolov8n.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading AI model: {e}")

# --- TEXT TO SPEECH FUNCTION ---
def speak(text):
    if text and text != "Nothing detected":
        try:
            full_text = f"I can see {text}"
            tts = gTTS(text=full_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except Exception as e:
            st.warning("Audio generation failed, but detection is ready.")

# --- UI LAYOUT ---
st.title("🤖 AI Vision & Voice Assistant")
st.write("Computer Vision application that detects objects and speaks the results.")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Choose Input Mode:", ("Upload Image", "Image URL", "Live Camera"))

img_input = None

if mode == "Upload Image":
    img_input = st.file_uploader("Select an image...", type=['jpg', 'jpeg', 'png'])

elif mode == "Image URL":
    url = st.text_input("Enter Image URL:")
    if url:
        img_input = url

elif mode == "Live Camera":
    img_input = st.camera_input("Take a photo to analyze")

# --- MAIN LOGIC ---
if img_input:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Selected Image")
        st.image(img_input, use_container_width=True)

    if st.button("🚀 Run AI Analysis"):
        with st.spinner('Analyzing... please wait.'):
            # Object Detection
            results = model(img_input)
            
            # Process results
            res_plotted = results[0].plot()  # Annotated image (BGR)
            # Convert BGR (OpenCV) to RGB (PIL/Streamlit)
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            # Get Labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            unique_labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("AI Detection Result")
                st.image(res_image, use_container_width=True)
                st.success(f"**Objects Found:** {unique_labels}")
                
                # Convert Result to Speech
                speak(unique_labels)

st.sidebar.markdown("---")
st.sidebar.info("Using YOLOv8 Nano (Free) & gTTS. No API keys needed.")
