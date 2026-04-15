import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Page setting
st.set_page_config(page_title="AI Vision", layout="centered")

st.title("🔍 AI Object Detector")
st.write("Upload an image to detect objects using YOLOv11.")

# Model load (Cached)
@st.cache_resource
def load_model():
    # Ye line model download karegi, isme 1-2 mins lag sakte hain pehli dafa
    return YOLO("yolo11n.pt") 

try:
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Analyze'):
            with st.spinner('Detecting...'):
                results = model.predict(source=image, conf=0.4)
                res_plotted = results[0].plot()
                st.image(res_plotted, caption='Detected Objects', use_container_width=True)
                
                # List names
                names = results[0].names
                detected = [names[int(c)] for c in results[0].boxes.cls]
                st.success(f"Found: {', '.join(set(detected)) if detected else 'No objects'}")

except Exception as e:
    st.error(f"Error loading model: {e}")
