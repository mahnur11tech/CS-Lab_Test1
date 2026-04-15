import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import tempfile
import requests
from io import BytesIO

# --- PAGE SETUP ---
st.set_page_config(page_title="Smart AI Vision", layout="wide")
st.title("🤖 Smart AI Vision Assistant")

@st.cache_resource
def load_model():
    # Smartest lightweight model for object detection
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

def speak(text):
    if text:
        try:
            tts = gTTS(text=f"I can see {text}", lang='en')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                st.audio(fp.name, format="audio/mp3")
        except:
            pass

# --- UI ---
option = st.sidebar.radio("Image Source:", ["Upload File", "Image URL", "Camera"])
img = None

if option == "Upload File":
    file = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("Paste Image URL here:")
    if url:
        try:
            # Adding headers to mimic a browser (Fixes most URL issues)
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("URL se image load nahi ho saki. Dusra link try karein.")

elif option == "Camera":
    st.info("💡 Camera access allow karne ke liye browser bar mein 'Lock' 🔒 icon par click karein.")
    cam_file = st.camera_input("Take a photo")
    if cam_file: img = Image.open(cam_file).convert("RGB")

if img:
    # Resize image for better AI performance (not too big, not too small)
    img.thumbnail((800, 800)) 
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    if st.button("🚀 Analyze Now"):
        with st.spinner("AI is analyzing..."):
            results = detector(img)
            
            draw = ImageDraw.Draw(img)
            found_items = []

            for res in results:
                # Confidence threshold (0.8 means 80% sure)
                if res['score'] > 0.8:
                    label = res['label']
                    box = res['box']
                    found_items.append(label)

                    # Draw thick box
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="yellow", width=5)
                    # Label text
                    draw.text((box['xmin'], box['ymin'] - 20), label.upper(), fill="yellow")

            with col2:
                st.subheader("AI Result")
                st.image(img, use_container_width=True)
                
                if found_items:
                    unique_items = list(set(found_items))
                    detected_text = ", ".join(unique_items)
                    st.success(f"Detected: {detected_text}")
                    speak(detected_text)
                else:
                    st.warning("AI kuch nahi pehchan saka. Confidence level zyada hai.")
