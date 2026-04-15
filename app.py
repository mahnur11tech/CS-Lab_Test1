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
    # 'timm' library install hona zaroori hai is model ke liye
    return pipeline("object-detection", model="facebook/detr-resnet-50")

try:
    detector = load_model()
except Exception as e:
    st.error(f"Model load hone mein masla hai: {e}")

# --- VOICE FUNCTION ---
def speak(text):
    if text:
        try:
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            st.warning("Voice playback failed.")

# --- UI NAVIGATION ---
st.sidebar.header("Options")
option = st.sidebar.radio("Image kahan se layen?", ["Upload File", "Image URL", "Camera"])

img = None

if option == "Upload File":
    file = st.file_uploader("Image select karein", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("URL yahan paste karein:")
    if url:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("URL se image load nahi ho saki. Link check karein.")

elif option == "Camera":
    cam_file = st.camera_input("Photo khinchen")
    if cam_file: img = Image.open(cam_file).convert("RGB")

# --- ANALYSIS ---
if img:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)

    if st.button("Analyze & Speak"):
        with st.spinner("AI pehchan raha hai..."):
            results = detector(img)
            
            draw = ImageDraw.Draw(img)
            found_items = []

            for res in results:
                if res['score'] > 0.7:  # Sirf pakki cheezon ko dikhaye ga
                    label = res['label']
                    box = res['box']
                    found_items.append(label)

                    # Draw Box
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=6)
                    # Draw Label Text
                    draw.text((box['xmin'], box['ymin'] - 15), f"{label.upper()}", fill="red")

            with col2:
                st.subheader("AI Detection Result")
                st.image(img, use_container_width=True)
                
                if found_items:
                    detected_text = ", ".join(set(found_items))
                    st.success(f"Detected: {detected_text}")
                    speak(detected_text)
                else:
                    st.warning("Kuch nazar nahi aaya. Confidence level kam kar ke dekhen.")
