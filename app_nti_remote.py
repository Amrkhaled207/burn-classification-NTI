
import os
import hashlib
import tempfile
import streamlit as st
import numpy as np
import cv2
import requests
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(page_title="🔥 Burn Severity Classifier", layout="centered")

# ---------- Config ----------
DEFAULT_MODEL_NAME = "model.h5"
CACHE_DIR = Path(os.environ.get("MODEL_CACHE_DIR", Path.home() / ".cache" / "nti_burn_model"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = CACHE_DIR / DEFAULT_MODEL_NAME

def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def stream_download(url: str, dst: Path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        chunk = 1024 * 1024
        with open(dst, "wb") as f:
            prog = st.progress(0.0, text="Downloading model...")
            downloaded = 0
            for data in r.iter_content(chunk_size=chunk):
                if data:
                    f.write(data)
                    downloaded += len(data)
                    if total:
                        prog.progress(min(downloaded / total, 1.0), text=f"Downloading model... {downloaded/1e6:.1f}/{total/1e6:.1f} MB")
            prog.progress(1.0, text="Download complete.")

@st.cache_resource(show_spinner=True)
def load_or_download_model() -> Path:
    # 1) If a local model exists in repo (for local dev), use it
    local_repo_model = Path("model.h5")
    if local_repo_model.exists():
        return local_repo_model

    # 2) Otherwise, download from URL in secrets
    url = st.secrets.get("MODEL_URL", "").strip()
    expected_sha256 = st.secrets.get("MODEL_SHA256", "").strip()

    if not url:
        st.error("No model found and no MODEL_URL provided in secrets. Add MODEL_URL in app settings.")
        st.stop()

    tmp_path = CACHE_DIR / (DEFAULT_MODEL_NAME + ".partial")
    try:
        stream_download(url, tmp_path)
        tmp_path.rename(MODEL_PATH)
    except Exception as e:
        st.exception(e)
        st.stop()

    if expected_sha256:
        actual = sha256_of_file(MODEL_PATH)
        if actual.lower() != expected_sha256.lower():
            st.error("Model checksum failed. Please verify MODEL_SHA256 in secrets.")
            st.stop()

    return MODEL_PATH

@st.cache_resource(show_spinner=True)
def load_burn_model(model_file: Path):
    return load_model(str(model_file))

@st.cache_data(show_spinner=False)
def preprocess_image(file_bytes, target_size=(128, 128)):
    image_data = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    image_bgr = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode image. Please upload a valid JPG/PNG.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, target_size)
    normalized = resized / 255.0
    batched = np.expand_dims(normalized, axis=0).astype("float32")
    return resized, batched

def main():
    st.title("🔥 Burn Severity Classifier (Remote Model)")
    st.caption("The model is downloaded at startup using a private URL in Streamlit secrets.")

    model_file = load_or_download_model()
    model = load_burn_model(model_file)

    uploaded = st.file_uploader("📷 Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("👈 Please upload a JPG or PNG image to continue.")
        return

    img_preview, input_tensor = preprocess_image(uploaded.read())
    st.image(img_preview, caption="🖼️ Uploaded Image", use_column_width=True)

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
