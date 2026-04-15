import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import tempfile
import requests
from io import BytesIO

# --- PAGE SETUP ---
st.set_page_config(page_title="Smart AI Vision", layout="wide")
st.title("🤖 AI Vision: Identify & Speak")

# --- LOAD AI MODEL (Cached) ---
@st.cache_resource
def load_model():
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

# --- VOICE FUNCTION ---
def speak(text):
    if text:
        try:
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            st.warning("Audio generate nahi ho saka.")

# --- UI NAVIGATION ---
option = st.sidebar.radio("Image Source Select Karein:", ["Upload File", "Image URL", "Camera"])

img = None

if option == "Upload File":
    file = st.file_uploader("Image select karein", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("Image ka URL yahan paste karein:")
    if url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("URL sahi nahi hai ya image load nahi ho saki.")

elif option == "Camera":
    cam_file = st.camera_input("Photo khinchen")
    if cam_file: img = Image.open(cam_file).convert("RGB")

# --- ANALYSIS LOGIC ---
if img:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner("AI Analysis kar raha hai..."):
            results = detector(img)
            
            draw = ImageDraw.Draw(img)
            # Font size set karna (agar available ho, warna default)
            try:
                font = ImageFont.load_default()
            except:
                font = None

            found_items = []

            for res in results:
                if res['score'] > 0.8:
                    label = res['label']
                    box = res['box']
                    found_items.append(label)

                    # Draw Box
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=5)
                    # Draw Text (Bird, Dog, etc.)
                    draw.text((box['xmin'], box['ymin'] - 15), f"{label}", fill="red", font=font)

            with col2:
                st.subheader("AI Detection Result")
                st.image(img, use_container_width=True)
                
                if found_items:
                    detected_text = ", ".join(set(found_items))
                    st.success(f"Detected: {detected_text}")
                    # Voice Output
                    speak(detected_text)
                else:
                    st.warning("AI ko kuch nazar nahi aaya.")
