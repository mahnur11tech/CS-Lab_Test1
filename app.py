import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from gtts import gTTS
import tempfile

# --- PAGE SETUP ---
st.set_page_config(page_title="Smart AI Vision", layout="wide")
st.title("🤖 AI Vision: URL & Camera Fix")

# --- LOAD AI MODEL ---
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
            pass

# --- UI ---
option = st.sidebar.radio("Image Source:", ["Image URL", "Upload File", "Camera"])
img = None

if option == "Image URL":
    url = st.text_input("Yahan Image ka Direct Link paste karein (e.g. .jpg ya .png):")
    if url:
        try:
            # Headers add kiye hain taaki URL download fail na ho
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            st.error(f"URL se image nahi mil saki. Error: {e}")

elif option == "Upload File":
    file = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Camera":
    cam_file = st.camera_input("Take a photo")
    if cam_file: img = Image.open(cam_file).convert("RGB")

# --- ANALYSIS ---
if img:
    # Image ko thoda resize karna detection behtar banata hai
    img.thumbnail((800, 800))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(img, use_container_width=True)

    if st.button("🚀 Analyze Now"):
        with st.spinner("AI pehchan raha hai..."):
            # Model run karein
            results = detector(img)
            
            draw = ImageDraw.Draw(img)
            found_items = []

            for res in results:
                # Agar URL wali image pehchan nahi raha toh confidence thoda kam (0.6) rakhein
                if res['score'] > 0.6: 
                    label = res['label']
                    box = res['box']
                    found_items.append(label)

                    # Bounding Box Draw karein
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="yellow", width=6)
                    draw.text((box['xmin'], box['ymin'] - 15), label.upper(), fill="yellow")

            with col2:
                st.subheader("AI Detection Result")
                st.image(img, use_container_width=True)
                
                if found_items:
                    detected_text = ", ".join(list(set(found_items)))
                    st.success(f"Detected: {detected_text}")
                    speak(detected_text)
                else:
                    st.warning("AI ko is image mein kuch khas nazar nahi aaya. Dusri image try karein.")
