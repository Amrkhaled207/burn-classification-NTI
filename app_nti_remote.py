import streamlit as st
import numpy as np
import cv2
import requests
from pathlib import Path
from tensorflow.keras.models import load_model
from PIL import Image

# ==============================
# Settings
# ==============================
MODEL_PATH = Path("model.h5")
MODEL_URL = st.secrets["MODEL_URL"]  # set in Streamlit Cloud Secrets
MODEL_SHA256 = st.secrets.get("MODEL_SHA256")  # optional checksum for integrity

# ==============================
# Model Download Logic
# ==============================
def download_model():
    """Download the model file if not already present."""
    if MODEL_PATH.exists():
        return
    st.info("Downloading model, please wait...")
    r = requests.get(MODEL_URL, stream=True)
    r.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    st.success("Model downloaded successfully!")

    # Optional integrity check
    if MODEL_SHA256:
        import hashlib
        sha256 = hashlib.sha256(open(MODEL_PATH, "rb").read()).hexdigest()
        if sha256 != MODEL_SHA256:
            raise ValueError("Model file checksum does not match!")

# ==============================
# Load Model
# ==============================
@st.cache_resource(show_spinner=True)
def load_burn_model():
    download_model()
    return load_model(str(MODEL_PATH))

# ==============================
# Image Preprocessing
# ==============================
@st.cache_data(show_spinner=False)
def preprocess_image(file_bytes, target_size=(128, 128)):
    image_data = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, target_size)
    normalized = resized / 255.0
    batched = np.expand_dims(normalized, axis=0).astype("float32")
    return resized, batched

# ==============================
# Main App
# ==============================
def main():
    st.set_page_config(page_title="🔥 Burn Severity Classifier", layout="centered")
    st.title("🔥 Burn Severity Classifier")
    st.caption("Upload a skin burn image to classify it as First, Second, or Third degree.")

    uploaded = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("👈 Please upload a JPG or PNG image to continue.")
        return

    model = load_burn_model()
    img_preview, input_tensor = preprocess_image(uploaded.read())

    st.image(img_preview, caption="🖼️ Uploaded Image", use_column_width=True)
    st.markdown("---")
    st.subheader("📊 Prediction")

    with st.spinner("Analyzing burn severity..."):
        preds = model.predict(input_tensor)[0]
        idx = int(np.argmax(preds))
        confidence = float(preds[idx])

    labels = ["First Degree", "Second Degree", "Third Degree"]
    label = labels[idx]

    st.success(f"**{label}** detected with **{confidence*100:.2f}%** confidence.")

    st.markdown("### 🔬 Prediction Confidence")
    st.bar_chart({
        "Confidence": {
            "First Degree": float(preds[0]),
            "Second Degree": float(preds[1]),
            "Third Degree": float(preds[2]),
        }
    })

if __name__ == "__main__":
    main()
