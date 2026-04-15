import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from gtts import gTTS
import tempfile
import cv2
import numpy as np
import os

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="Pro AI Vision", layout="wide", page_icon="🚀")

# Custom CSS for Styling
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    /* Title Style */
    .main-title {
        font-size: 45px;
        font-weight: 800;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px #000000;
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
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
        transform: scale(1.02);
    }
    /* Result Boxes */
    .stSuccess, .stWarning {
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-title">🚀 Pro AI Vision Assistant</p>', unsafe_allow_html=True)
st.write("<p style='text-align: center; color: #8b949e;'>Identify Objects in Images, URLs, and Videos with Voice Feedback</p>", unsafe_allow_html=True)

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

def speak(text):
    if text:
        try:
            tts = gTTS(text=f"I see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except: pass

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Input Source", ["Upload Image", "Image URL", "Camera", "Upload Video"])

img = None

# --- 4. INPUT HANDLING ---
if option == "Upload Image":
    file = st.file_uploader("Choose a file", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("🔗 Paste Direct Image URL:")
    if url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except: st.error("❌ Could not load image from URL.")

elif option == "Camera":
    st.info("📸 Click below
