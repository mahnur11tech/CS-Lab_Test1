import os
import tempfile
import streamlit as st
from PIL import Image
from gtts import gTTS
from ultralytics import YOLO

# --- OS SETTINGS FOR STABILITY ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Vision Assistant",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3.5em; background-color: #ff4b4b; color: white; font-weight: bold; border: none; }
    .stButton>button:hover { background-color: #ff3333; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Smart AI Vision & Voice Assistant")
st.write("This app uses a pre-trained **YOLOv8** model to detect objects and speak them out loud.")

# --- MODEL LOADING (MEMORY EFFICIENT) ---
@st.cache_resource
def load_model():
    # 'yolov8n.pt' is the nano model (approx 6MB). Do NOT use larger models on Cloud.
    return YOLO('yolov8n.pt')

model = load_model()

def speak(text):
    """Converts detection labels to speech"""
    if text and text != "Nothing detected":
        try:
            full_text = f"I can see {text}"
            tts = gTTS(text=full_text, lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except Exception as e:
            st.error("Audio error. Detection is still visible above.")

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("🔧 Settings")
mode = st.sidebar.radio("Input Method:", ("Upload File", "Image URL", "Live Camera"))

img_input = None

if mode == "Upload File":
    img_input = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

elif mode == "Image URL":
    url = st.text_input("Enter Image URL:")
    if url:
        img_input = url

elif mode == "Live Camera":
    img_input = st.camera_input("Take a photo")

# --- MAIN LOGIC ---
if img_input:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📸 Original Image")
        st.image(img_input, use_container_width=True)

    if st.button("🚀 Analyze & Identify"):
        with st.spinner('AI is processing...'):
            # Run YOLO Detection
            results = model(img_input)
            
            # Process Result Image (Convert BGR to RGB)
            res_plotted = results[0].plot()
            res_image = Image.fromarray(res_plotted[..., ::-1])
            
            # Get Detected Labels
            names = [model.names[int(c)] for c in results[0].boxes.cls]
            unique_labels = ", ".join(set(names)) if names else "Nothing detected"

            with col2:
                st.subheader("🔍 AI Detection")
                st.image(res_image, use_container_width=True)
                st.success(f"**Objects Found:** {unique_labels}")
                
                # Audio Feedback
                speak(unique_labels)

st.sidebar.markdown("---")
st.sidebar.info("Tip: If the app crashes, try a smaller image size. Streamlit Cloud has 1GB RAM limit.")
