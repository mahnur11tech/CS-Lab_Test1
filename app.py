import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw

# --- PAGE CONFIG ---
st.set_page_config(page_title="Simple AI Detector", layout="centered")

st.title("🤖 2-File Smart Detector")
st.write("This version is designed to run without 'packages.txt'.")

# --- LOAD MODEL (Hugging Face) ---
@st.cache_resource
def load_model():
    # Using a super light-weight object detection model
    return pipeline("object-detection", model="facebook/detr-resnet-50")

detector = load_model()

# --- UI ---
img_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Your Image", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("AI is looking..."):
            # Run Detection
            results = detector(img)
            
            # Draw on image using PIL (No OpenCV needed!)
            draw = ImageDraw.Draw(img)
            found_objects = []

            for res in results:
                box = res['box']
                label = res['label']
                score = res['score']
                
                if score > 0.7:  # Only show confident results
                    draw.rectangle([(box['xmin'], box['ymin']), (box['xmax'], box['ymax'])], outline="red", width=3)
                    draw.text((box['xmin'], box['ymin']), f"{label} {int(score*100)}%", fill="red")
                    found_objects.append(label)

            # Show Result
            st.image(img, caption="AI Detection Result", use_container_width=True)
            
            if found_objects:
                st.success(f"I detected: {', '.join(set(found_objects))}")
            else:
                st.warning("I couldn't identify anything clearly.")
