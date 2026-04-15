import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw

# --- PAGE SETUP ---
st.set_page_config(page_title="Easy AI Detector", layout="centered")

st.title("🤖 2-File Smart Detector")
st.write("Ye app bina kisi heavy library ke chalti hai.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Ye model light hai aur bina OpenCV ke chalta hai
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

# --- UI ---
img_file = st.file_uploader("Image upload karein", type=['jpg', 'png', 'jpeg'])

if img_file:
    img = Image.open(img_file).convert("RGB")
    st.image(img, caption="Original Image", use_container_width=True)

    if st.button("Check Karein"):
        with st.spinner("AI dekh raha hai..."):
            results = detector(img)
            
            # Draw using PIL (No OpenCV needed)
            draw = ImageDraw.Draw(img)
            for res in results:
                if res['score'] > 0.8:
                    box = res['box']
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=4)
                    draw.text((box['xmin'], box['ymin']), f"{res['label']}", fill="red")

            st.image(img, caption="AI Result", use_container_width=True)
            st.success("Analysis Mukammal!")
