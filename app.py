
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="AI Vision Pro", layout="centered")

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Description
st.title("🔍 AI Object Detector")
st.write("Upload an image, and our AI will detect objects for you—No API, No Fees!")

# Function to load model (Cached to avoid reloading every time)
@st.cache_resource
def load_model():
    # 'yolo11n.pt' is a lightweight model that runs fast without a GPU
    return YOLO("yolo11n.pt") 

model = load_model()

# Sidebar for Settings
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.4)

# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Process Button
    if st.button('Analyze Image'):
        with st.spinner('AI is thinking...'):
            # Perform Detection
            results = model.predict(source=image, conf=confidence)
            
            # Extract the plotted image (with boxes)
            res_plotted = results[0].plot()
            
            # Display Result
            st.subheader("Detection Results")
            st.image(res_plotted, caption='Detected Objects', use_container_width=True)
            
            # Show Detected Object List
            names = results[0].names
            detected_classes = [names[int(c)] for c in results[0].boxes.cls]
            
            if detected_classes:
                st.success(f"Detected: {', '.join(set(detected_classes))}")
            else:
                st.warning("No objects detected with current confidence.")

st.divider()
st.caption("Powered by YOLOv11 and Streamlit. Runs locally on your server.")
