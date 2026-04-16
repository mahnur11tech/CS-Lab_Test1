import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import tempfile
import requests
from io import BytesIO
import cv2
import numpy as np
import os

# --- 1. PAGE CONFIG & MODERN STYLE ---
st.set_page_config(page_title="Pro AI Vision Assistant", layout="wide", page_icon="🤖")

st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 5px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #00d4ff;
        color: black;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #008fb3;
        color: white;
        transform: translateY(-2px);
    }
    section[data-testid="stSidebar"] {
        background-color: #111b21;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    # Object detection model (Fast and Accurate)
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

def speak(text):
    if text:
        try:
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except: pass

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🎮 AI Control Panel")
option = st.sidebar.radio("Input Source Chunain:", ["Upload Image", "Image URL", "Camera", "Upload Video"])

img = None

# --- 4. HANDLING INPUTS ---

if option == "Upload Image":
    file = st.file_uploader("Image select karein", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("🔗 Image ka Link (URL) yahan paste karein:")
    if url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except: st.error("Link sahi nahi hai ya image load nahi ho saki.")

elif option == "Camera":
    st.markdown("### 📸 Live Capture")
    st.info("💡 Agar niche 'Take Photo' button nahi aa raha, to Browser bar mein Lock (🔒) par click kar ke Camera Allow karein aur page Refresh karein.")
    # 'key' add karne se widget restart ho jata hai
    cam_file = st.camera_input("Apni photo khinchen", key="ai_camera_widget")
    if cam_file: img = Image.open(cam_file).convert("RGB")

elif option == "Upload Video":
    v_file = st.file_uploader("Video select karein (.mp4 recommended)", type=['mp4', 'mov', 'avi'])
    if v_file:
        st.video(v_file)
        if st.button("🔍 Analyze Video Frames"):
            with st.spinner("Video scan ho rahi hai..."):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(v_file.read())
                cap = cv2.VideoCapture(tfile.name)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Frames to scan (Start, Middle, End)
                frames_to_check = [0, frame_count//2, frame_count-5]
                found_v = []
                
                for f_idx in frames_to_check:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        pil_f = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        res_v = detector(pil_f)
                        found_v.extend([r['label'] for r in res_v if r['score'] > 0.7])
                
                cap.release()
                os.unlink(tfile.name)
                
                if found_v:
                    detected_v = ", ".join(list(set(found_v)))
                    st.success(f"Video Detection Result: {detected_v}")
                    speak(detected_v)
                else: st.warning("Video mein kuch detect nahi hua.")

# --- 5. IMAGE DETECTION LOGIC ---
st.markdown('<p class="main-title">🤖 AI Vision Assistant</p>', unsafe_allow_html=True)

if img and option != "Upload Video":
    # Resize for better processing
    img.thumbnail((800, 800))
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🖼️ Original View")
        st.image(img, use_container_width=True)

    if st.button("🚀 RUN AI SCAN"):
        with st.spinner("AI pehchan raha hai..."):
            results = detector(img)
            draw = ImageDraw.Draw(img)
            found_items = []

            for res in results:
                if res['score'] > 0.7:
                    label = res['label']
                    box = res['box']
                    found_items.append(label)
                    # Drawing stylish box
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="#00d4ff", width=6)
                    draw.text((box['xmin'], box['ymin']-15), label.upper(), fill="#00d4ff")

            with col2:
                st.markdown("### 🎯 Detection Result")
                st.image(img, use_container_width=True)
                if found_items:
                    final_text = ", ".join(list(set(found_items)))
                    st.success(f"**Nazar Aaya:** {final_text}")
                    speak(final_text)
                else:
                    st.warning("AI ko kuch samajh nahi aaya.")

st.sidebar.markdown("---")
st.sidebar.caption("Smart AI Vision Tool | v2.0")
