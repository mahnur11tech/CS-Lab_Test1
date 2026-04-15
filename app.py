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

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Vision & Video Assistant", layout="wide")
st.title("🤖 AI Vision: Image & Video Detector")

# --- LOAD AI MODEL ---
@st.cache_resource
def load_model():
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

# --- UI NAVIGATION ---
option = st.sidebar.radio("Select Source:", ["Upload Image", "Image URL", "Camera", "Upload Video"])

img = None

if option == "Upload Image":
    file = st.file_uploader("Select Image", type=['jpg', 'png', 'jpeg'])
    if file: img = Image.open(file).convert("RGB")

elif option == "Image URL":
    url = st.text_input("Paste Image URL:")
    if url:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=15)
            img = Image.open(BytesIO(response.content)).convert("RGB")
        except: st.error("URL error!")

elif option == "Camera":
    st.warning("⚠️ Agar camera nahi chal raha, to Browser bar mein Lock 🔒 icon se permission 'Allow' karein.")
    cam_file = st.camera_input("Take a photo")
    if cam_file: img = Image.open(cam_file).convert("RGB")

elif option == "Upload Video":
    video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
    if video_file:
        st.video(video_file)
        if st.button("Analyze Video"):
            with st.spinner("Video process ho rahi hai (Frames analysis)..."):
                # Temporary file save karein
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Sirf 3 main frames uthayen (Start, Middle, End) taakay memory crash na ho
                frames_to_check = [0, frame_count//2, frame_count-5]
                all_found = []
                
                for f_idx in frames_to_check:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        res = detector(pil_img)
                        for r in res:
                            if r['score'] > 0.7:
                                all_found.append(r['label'])
                
                cap.release()
                os.unlink(tfile.name) # Temp file delete karein
                
                if all_found:
                    unique_items = ", ".join(list(set(all_found)))
                    st.success(f"Video mein ye cheezain mili: {unique_items}")
                    speak(unique_items)
                else:
                    st.warning("Video mein kuch khas nahi mila.")

# --- IMAGE ANALYSIS LOGIC ---
if img and option != "Upload Video":
    col1, col2 = st.columns(2)
    with col1: st.image(img, caption="Input Image", use_container_width=True)

    if st.button("Analyze Now"):
        with st.spinner("AI is thinking..."):
            results = detector(img)
            draw = ImageDraw.Draw(img)
            found = []
            for res in results:
                if res['score'] > 0.7:
                    label = res['label']
                    box = res['box']
                    found.append(label)
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=5)
            
            with col2:
                st.image(img, caption="AI Detection Result", use_container_width=True)
                if found:
                    detected_text = ", ".join(list(set(found)))
                    st.success(f"Detected: {detected_text}")
                    speak(detected_text)
                else: st.warning("Kuch nahi mila.")
