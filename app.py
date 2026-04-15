import os
import subprocess
import sys

# Force install requirements from your specific file name
def install_packages():
    try:
        import ultralytics
    except ImportError:
        # Agar library nahi mil rahi, to ye command usay install karegi
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirement.txt"])

# Installation function ko call karen
install_packages()

# --- Baqi saara code yahan se shuru hoga ---
import streamlit as st
from ultralytics import YOLO
from PIL import Image
# ... (baqi purana code)
